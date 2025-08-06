# livechat.py
import asyncio
import logging
import random
from datetime import datetime, timezone
from collections import deque

from curl_cffi.requests import AsyncSession

# Import our refactored clients and data model
from data_models import UnifiedMessage
from general_utils import get_env_var
from youtube import YTLive, YouTubeApiError
from twitch import Bot as TwitchBot, update_owner_bot_ids, CLIENT_ID, CLIENT_SECRET, BOT_ID, OWNER_ID
from kick import KickClient, KickApiError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')

class LiveChatController:
    """
    An asyncio-based controller to manage and fetch messages from multiple live chat platforms.
    """
    def __init__(self, fetch_youtube=False, fetch_twitch=False, fetch_kick=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._fetch_youtube = fetch_youtube
        self._fetch_twitch = fetch_twitch
        self._fetch_kick = fetch_kick

        # Central pool for randomly selecting a message
        self._all_messages = []
        # Dedicated dequeue for the Twitch bot to push messages to
        twitch_max_messages = get_env_var("TWITCH_MAX_MESSAGES", var_type=int)
        self.twitch_messages: deque[UnifiedMessage] = deque(maxlen=twitch_max_messages)

        # Asynchronous clients
        self.http_session: AsyncSession | None = None
        self.youtube: YTLive | None = None
        self.kick: KickClient | None = None
        self.twitch_bot: TwitchBot | None = None
        self.twitch_task: asyncio.Task | None = None

        # State management for polling clients
        self.yt_next_page_token: str | None = get_env_var("LAST_NEXT_PAGE_TOKEN", var_type=str)
        self.kick_last_timestamp = datetime.now(timezone.utc)

        self.logger.info(f"Controller initialized with settings: YouTube={fetch_youtube}, Twitch={fetch_twitch}, Kick={fetch_kick}")

    @classmethod
    def create(cls):
        """Factory method to create a controller based on environment variables."""
        fetch_youtube = get_env_var("YT_FETCH", var_type=bool)
        fetch_twitch = get_env_var("TW_FETCH", var_type=bool)
        fetch_kick = get_env_var("KI_FETCH", var_type=bool)

        if not any([fetch_youtube, fetch_twitch, fetch_kick]):
            logging.warning("No fetch variables are set. LiveChatController will not fetch from any platform.")
            return None

        return cls(fetch_youtube=fetch_youtube, fetch_twitch=fetch_twitch, fetch_kick=fetch_kick)

    async def setup_clients(self):
        """Initializes and prepares all enabled clients for operation."""
        self.logger.info("Setting up clients...")
        self.http_session = AsyncSession()

        if self._fetch_youtube:
            try:
                self.youtube = YTLive()
                await self.youtube.setup()
            except (YouTubeApiError, ValueError, FileNotFoundError) as e:
                self.logger.error(f"Failed to setup YouTube client: {e}")
                self.youtube = None # Disable on failure

        if self._fetch_kick:
            kick_channel = get_env_var("KI_CHANNEL")
            if kick_channel and self.http_session:
                self.kick = KickClient(username=kick_channel, session=self.http_session)
            else:
                self.logger.error("KI_CHANNEL not set, cannot initialize Kick client.")
                self.kick = None

        if self._fetch_twitch:
            try:
                owner_id, bot_id = await update_owner_bot_ids()
                self.twitch_bot = TwitchBot(
                    message_list=self.twitch_messages,
                    client_id=CLIENT_ID,
                    client_secret=CLIENT_SECRET,
                    bot_id=bot_id,
                    owner_id=owner_id,
                    prefix="!",
                    logger=None
                )
            except Exception as e:
                self.logger.error(f"Failed to setup Twitch bot: {e}")
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
        return None

### The section below demonstrates how to use the controller.
async def main():
    """Main execution function."""
    from dotenv import load_dotenv
    import time
    load_dotenv()

    print("--- Live Chat Controller ---")
    controller = LiveChatController.create()
    if not controller:
        print("Controller could not be created. Check your .env configuration. Exiting.")
        return

    await controller.setup_clients()
    await controller.start_services()

    try:
        # Example loop to fetch a message every 15 seconds
        for i in range(10): # Run 10 times for this example
            print(f"\n--- Fetch Cycle {i+1} ---")
            message = await controller.fetch_chat_message()
            if message:
                print(f"--> Winner: [{message.platform}] {message.username}: {message.content}")
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