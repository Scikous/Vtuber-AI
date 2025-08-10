# livechat.py
import asyncio
import logging
import random
from datetime import datetime, timezone
from collections import deque
from typing import List, Optional, Tuple
from curl_cffi.requests import AsyncSession

# Import our refactored clients and data model
from data_models import UnifiedMessage
from general_utils import get_env_var
from youtube import YTLive, YouTubeApiError
from twitch import Bot as TwitchBot, fetch_twitch_user_ids#update_owner_bot_ids, CLIENT_ID, CLIENT_SECRET, BOT_ID, OWNER_ID
from kick import KickClient


logger = logging.getLogger(__name__)

class LiveChatController:
    """
    An asyncio-based controller to manage and fetch messages from multiple live chat platforms.
    """
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        # Unpack configuration
        self.youtube_config = self.config.get('youtube', {})
        self.twitch_config = self.config.get('twitch', {})
        self.kick_config = self.config.get('kick', {})

        self._fetch_youtube = self.youtube_config.get('enabled', False)
        self._fetch_twitch = self.twitch_config.get('enabled', False)
        self._fetch_kick = self.kick_config.get('enabled', False)

        # Central pool for randomly selecting a message
        self._all_messages: List[UnifiedMessage] = []
        # Dedicated deque for the Twitch bot to push messages to
        twitch_max_len = self.twitch_config.get('max_messages', 100)
        self.twitch_messages: deque[UnifiedMessage] = deque(maxlen=twitch_max_len)
        
        # State management for polling clients
        self.yt_next_page_token: Optional[str] | None = get_env_var("LAST_NEXT_PAGE_TOKEN", var_type=str)
        if self._fetch_kick:
            self.kick_last_timestamp = datetime.now(timezone.utc)

        # Asynchronous clients
        self.http_session: Optional[AsyncSession] = None
        self.youtube: Optional[YTLive] = None
        self.kick: Optional[KickClient] = None
        self.twitch_bot: Optional[TwitchBot] = None
        self.twitch_task: Optional[asyncio.Task] = None

        self.logger.info(f"Controller initialized with settings: YouTube={self._fetch_youtube}, Twitch={self._fetch_twitch}, Kick={self._fetch_kick}")

    async def setup_clients(self):
        """Initializes and prepares all enabled clients for operation."""
        self.logger.info("Setting up clients...")

        if self._fetch_youtube:
            try:
                self.youtube = YTLive(
                    channel_id=self.youtube_config['channel_id'],
                    client_secret_file=self.youtube_config['client_secret_file']
                )
                await self.youtube.setup()
            except (YouTubeApiError, ValueError, FileNotFoundError) as e:
                self.logger.error(f"Failed to setup YouTube client: {e}")
                self.youtube = None # Disable on failure

        if self._fetch_kick:
            self.http_session = AsyncSession()
            kick_channel = self.kick_config.get('channel_name')
            if kick_channel and self.http_session:
                self.kick = KickClient(username=kick_channel, session=self.http_session)
            else:
                self.logger.error("Kick channel_name not in config, cannot initialize client.")
                self.kick = None

        if self._fetch_twitch:
            try:
                # Ensure all required Twitch config keys are present
                required_keys = ['client_id', 'client_secret', 'bot_id', 'owner_id']
                if not all(key in self.twitch_config for key in required_keys):
                    raise ValueError(f"Missing one or more required Twitch config keys: {required_keys}")

                self.twitch_bot = TwitchBot(
                    message_list=self.twitch_messages,
                    bot_id=self.twitch_config['bot_id'],
                    client_id=self.twitch_config['client_id'],
                    client_secret=self.twitch_config['client_secret'],
                    owner_id=self.twitch_config['owner_id'],
                    prefix=self.twitch_config['prefix'] or '!'
                )
            except (Exception, ValueError) as e:
                self.logger.error(f"Failed to setup Twitch bot: {e}. Disabling.")
                self.twitch_bot = None

    async def start_services(self):
        """Starts any background services, like the Twitch bot."""
        if self.twitch_bot:
            self.logger.info("Starting Twitch bot background service...")
            self.twitch_task = asyncio.create_task(self.twitch_bot.start())

    async def stop_services(self):
        """Gracefully stops all background services and closes sessions."""
        self.logger.info("Stopping services...")
        if self.twitch_task and not self.twitch_task.done():
            self.twitch_task.cancel()
        if self.twitch_bot:
            await self.twitch_bot.close()
        if self.http_session:
            await self.http_session.close()
        self.logger.info("All services stopped.")

    async def fetch_chat_message(self) -> UnifiedMessage | None:
        """
        Performs a single, on-demand fetch from all active platforms,
        pools the messages, and returns one at random.
        """
        self._all_messages.clear()

        # 1. Drain the message list from the continuously-running Twitch bot
        if self.twitch_bot:
            self._all_messages.extend(self.twitch_messages)
            self.twitch_messages.clear()

        # 2. Create concurrent fetching tasks for poll-based clients (YouTube, Kick)
        fetch_tasks = []
        if self.youtube:
            fetch_tasks.append(self.youtube.get_live_chat_messages(self.yt_next_page_token))
        if self.kick:
            fetch_tasks.append(self.kick.fetch_new_messages(self.kick_last_timestamp))

        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # 3. Process results from the concurrent fetches
            # YouTube result will be at index 0 if enabled
            yt_idx = 0 if self.youtube else -1
            if self.youtube and yt_idx < len(results):
                yt_result = results[yt_idx]
                if isinstance(yt_result, Exception):
                    self.logger.error(f"Error fetching from YouTube: {yt_result}")
                else:
                    raw_messages, self.yt_next_page_token = yt_result
                    for msg in raw_messages:
                        unified = UnifiedMessage(
                            platform='YouTube',
                            username=msg['authorDetails']['displayName'],
                            content=msg['snippet']['displayMessage'],
                            timestamp=datetime.now(timezone.utc) # Note: YT API doesn't provide a reliable timestamp per message
                        )
                        self._all_messages.append(unified)

            # Kick result will be at the next index
            kick_idx = (yt_idx + 1) if self.kick else -1
            if self.kick and kick_idx < len(results):
                kick_result = results[kick_idx]
                if isinstance(kick_result, Exception):
                    self.logger.error(f"Error fetching from Kick: {kick_result}")
                else:
                    kick_messages = kick_result
                    if kick_messages:
                        self.kick_last_timestamp = kick_messages[-1].timestamp
                        for msg in kick_messages:
                            unified = UnifiedMessage(
                                platform='Kick',
                                username=msg.user.username,
                                content=msg.content,
                                timestamp=msg.timestamp,
                                color=msg.user.color
                            )
                            self._all_messages.append(unified)

        # 4. Select a random message from the aggregated pool
        if self._all_messages:
            message = random.choice(self._all_messages)
            self._all_messages.remove(message) # Prevent re-picking in the same batch
            self.logger.info(f"PICKED MESSAGE: {message}")
            self.logger.info(f"{len(self._all_messages)} messages remaining in this batch.")
            return message, self._all_messages
        
        self.logger.info("No new messages found in this fetch cycle.")
        return None, None

### The section below demonstrates how to use the controller.
async def main():
    """Main execution function demonstrating controller usage."""
    # The parent application is responsible for setting up logging.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
    
    from dotenv import load_dotenv
    from general_utils import get_env_var
    load_dotenv()

    print("--- Live Chat Controller ---")
    
    # 1. Build the configuration dictionary from a source like .env files.
    # This is the responsibility of the application's entry point.
    config = {
        "youtube": {
            "enabled": get_env_var("YT_FETCH", bool, False),
            "channel_id": get_env_var("YT_CHANNEL_ID", str),
            "client_secret_file": get_env_var("YT_OAUTH2_JSON", str),
            "initial_page_token": get_env_var("LAST_NEXT_PAGE_TOKEN", str)
        },
        "twitch": {
            "enabled": get_env_var("TW_FETCH", bool, False),
            "channel_name": get_env_var("TW_CHANNEL", str),
            "bot_name": get_env_var("TW_BOT_NAME", str),
            "client_id": get_env_var("TW_CLIENT_ID", str),
            "client_secret": get_env_var("TW_CLIENT_SECRET", str),
            "bot_id": get_env_var("TW_BOT_ID", str), # May be empty on first run
            "owner_id": get_env_var("TW_OWNER_ID", str), # May be empty on first run
            "prefix": "!",
            "max_messages": get_env_var("TWITCH_MAX_MESSAGES", int, 100)
        },
        "kick": {
            "enabled": get_env_var("KI_FETCH", bool, False),
            "channel_name": get_env_var("KI_CHANNEL", str)
        }
    }

    # 2. Handle any first-time setup, like fetching Twitch IDs.
    if config["twitch"]["enabled"] and (not config["twitch"]["bot_id"] or not config["twitch"]["owner_id"]):
        print("Twitch Bot ID or Owner ID not found. Attempting to fetch them...")
        try:
            owner_id, bot_id = await fetch_twitch_user_ids(
                client_id=config["twitch"]["client_id"],
                client_secret=config["twitch"]["client_secret"],
                channel_name=config["twitch"]["channel_name"],
                bot_name=config["twitch"]["bot_name"]
            )
            config["twitch"]["owner_id"] = owner_id
            config["twitch"]["bot_id"] = bot_id
            print(f"Successfully fetched IDs. Please add TW_OWNER_ID={owner_id} and TW_BOT_ID={bot_id} to your .env file for future runs.")
        except Exception as e:
            print(f"FATAL: Could not fetch Twitch IDs. Twitch support will be disabled. Error: {e}")
            config["twitch"]["enabled"] = False

    # 3. Instantiate and run the controller.
    if not any([c.get('enabled') for c in config.values()]):
        print("All fetch services are disabled in the configuration. Exiting.")
        return

    controller = LiveChatController(config)
    await controller.setup_clients()
    await controller.start_services()

    try:
        for i in range(10):
            print(f"\n--- Fetch Cycle {i+1} ---")
            message, remaining = await controller.fetch_chat_message()
            if message:
                print(f"--> Winner: [{message.platform}] {message.username}: {message.content}")
                if remaining:
                    print(f"    ({len(remaining)} other messages were also received in this batch)")
            else:
                print("--> No message was selected.")
            
            print("Sleeping for 15 seconds...")
            await asyncio.sleep(15)
    except asyncio.CancelledError:
        print("Main task cancelled.")
    finally:
        print("Shutting down services...")
        await controller.stop_services()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down.")