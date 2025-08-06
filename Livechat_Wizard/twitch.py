# twitch.py
import asyncio
import json
import logging
import random
from typing import Any, List, Tuple
from collections import deque

import twitchio
from twitchio import authentication, eventsub
from twitchio.ext import commands
from dotenv import load_dotenv, set_key, find_dotenv

# Import our standardized data model
from data_models import UnifiedMessage
from general_utils import get_env_var

# Load environment variables
load_dotenv()
CHANNEL = get_env_var("TW_CHANNEL", var_type=str)
BOT_NAME = get_env_var("TW_BOT_NAME", var_type=str)
CLIENT_ID = get_env_var("TW_CLIENT_ID", var_type=str)
CLIENT_SECRET = get_env_var("TW_CLIENT_SECRET", var_type=str)
# These may be empty on first run, we will fetch them.
BOT_ID = get_env_var("TW_BOT_ID", var_type=str)
OWNER_ID = get_env_var("TW_OWNER_ID", var_type=str)

LOGGER: logging.Logger = logging.getLogger(__name__)

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = (hex_color or "#FFFFFF").lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)  # Default to white for invalid formats
    try:
        return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except ValueError:
        return (255, 255, 255)

class Bot(commands.Bot):
    def __init__(self, message_list: deque[UnifiedMessage], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # This list is shared with the LiveChatController
        self.message_list = message_list

    async def setup_hook(self) -> None:
        """The setup hook for the bot, responsible for subscribing to events."""
        # Add our General Commands Component...
        await self.add_component(GeneralCommands(self))

        # We attempt to load tokens for users who have authorized our app.
        with open(".tio.tokens.json", "rb") as fp:
            tokens = json.load(fp)

        for user_id in tokens:
            if user_id == BOT_ID:
                continue

            # Subscribe to chat for everyone we have a token...
            chat = eventsub.ChatMessageSubscription(broadcaster_user_id=user_id, user_id=BOT_ID)
            await self.subscribe_websocket(chat)

    async def event_ready(self) -> None:
        LOGGER.info(f"Logged in as: {self.user}")

    async def event_oauth_authorized(self, payload: authentication.UserTokenPayload) -> None:
        # Stores tokens in .tio.tokens.json by default; can be overriden to use a DB for example
        # Adds the token to our Client to make requests and subscribe to EventSub...
        await self.add_token(payload.access_token, payload.refresh_token)

        if payload.user_id == BOT_ID:
            return
        # Subscribe to chat for new authorizations...
        chat = eventsub.ChatMessageSubscription(broadcaster_user_id=payload.user_id, user_id=BOT_ID)
        await self.subscribe_websocket(chat)

class GeneralCommands(commands.Component):
    def __init__(self, bot: Bot) -> None:
        self.bot = bot
    """A component that holds various chat commands for the bot. Preserved for functionality."""
    @commands.command()
    async def hi(self, ctx: commands.Context) -> None:
        await ctx.reply(f"Hi {ctx.chatter}!")

    @commands.command()
    async def say(self, ctx: commands.Context, *, message: str) -> None:
        await ctx.send(message)

    @commands.command()
    async def add(self, ctx: commands.Context, left: int, right: int) -> None:
        await ctx.reply(f"{left} + {right} = {left + right}")

    @commands.command()
    async def choice(self, ctx: commands.Context, *choices: str) -> None:
        await ctx.reply(f"You provided {len(choices)} choices, I choose: {random.choice(choices)}")

    @commands.command(aliases=["thanks", "thank"])
    async def give(self, ctx: commands.Context, user: twitchio.User, amount: int, *, message: str | None = None) -> None:
        msg = f"with message: {message}" if message else ""
        await ctx.send(f"{ctx.chatter.mention} gave {amount} thanks to {user.mention} {msg}")
    @commands.Component.listener()
    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        """
        This event is triggered when a new chat message is received via EventSub.
        This is the core of our message fetching for Twitch.
        """
        # The actual message data is in the 'data' attribute of the payload
        
        # Transform the message into our UnifiedMessage format
        unified_msg = UnifiedMessage(
            platform='Twitch',
            username=payload.chatter.name,
            content=payload.text,
            timestamp=payload.timestamp, # Use the notification timestamp
            color=_hex_to_rgb(str(payload.color))
        )
        
        # Append the message to the shared list
        self.bot.message_list.append(unified_msg)
        LOGGER.info(f"Twitch message captured from {unified_msg.username}: {unified_msg.content}")

async def update_owner_bot_ids() -> Tuple[str, str]:
    """
    Fetches user IDs for the channel owner and the bot, updating .env if necessary.
    This is a crucial first-time setup utility.
    """
    global OWNER_ID, BOT_ID
    # Only run if one of the IDs is missing.
    if OWNER_ID and BOT_ID:
        return OWNER_ID, BOT_ID

    print("OWNER_ID or BOT_ID not found in .env, fetching from Twitch API...")
    try:
        async with twitchio.Client(client_id=CLIENT_ID, client_secret=CLIENT_SECRET) as client:
            await client.login()
            users = await client.fetch_users(logins=[CHANNEL, BOT_NAME])
            
            owner_id = users[0].id
            bot_id = users[1].id

            dotenv_path = find_dotenv()
            set_key(dotenv_path, "TW_OWNER_ID", owner_id)
            set_key(dotenv_path, "TW_BOT_ID", bot_id)
            
            print(f"Updated .env: TW_OWNER_ID={owner_id}, TW_BOT_ID={bot_id}")
            return owner_id, bot_id
    except Exception as e:
        LOGGER.critical(f"Could not fetch owner/bot IDs. Please check TW_CHANNEL and TW_BOT_NAME. Error: {e}")
        raise

### The section below is for standalone testing.

async def _test_twitch_bot():
    """Main function to run the bot as a standalone application for testing."""
    print("--- Running Twitch Bot Standalone Test ---")
    # Use CRITICAL to avoid leaking tokens in logs during normal operation.
    # Use INFO for debugging setup issues.
    twitchio.utils.setup_logging(level=logging.INFO)

    try:
        owner_id, bot_id = await update_owner_bot_ids()
        
        # For testing, we create a dummy list to catch the messages.
        test_message_list: List[UnifiedMessage] = []
        
        bot_instance = Bot(
            message_list=test_message_list,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=bot_id,
            owner_id=owner_id,
            prefix="!",
            logger=None
        )
        
        # Start the bot and a task to print received messages.
        async def print_messages():
            while True:
                if test_message_list:
                    msg = test_message_list.pop(0)
                    print(f"  -> [TEST] Message received and captured: {msg}")
                await asyncio.sleep(1)

        await asyncio.gather(bot_instance.start(), print_messages())

    except Exception as e:
        LOGGER.error(f"An error occurred during standalone test: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(_test_twitch_bot())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down test due to KeyboardInterrupt.")