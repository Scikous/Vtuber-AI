# twitch.py
import asyncio
import json
import logging
import random
from typing import Any, List, Tuple, Deque
from collections import deque

import twitchio
from twitchio import authentication, eventsub
from twitchio.ext import commands

# Import our standardized data model
from data_models import UnifiedMessage

logger = logging.getLogger(__name__)

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
    def __init__(self, message_list: Deque[UnifiedMessage], **kwargs: Any) -> None:
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
            if user_id == self.bot_id:
                continue

            # Subscribe to chat for everyone we have a token...
            chat = eventsub.ChatMessageSubscription(broadcaster_user_id=user_id, user_id=self.bot_id)
            await self.subscribe_websocket(chat)

    async def event_ready(self) -> None:
        logger.info(f"Logged in as: {self.user}")

    async def event_oauth_authorized(self, payload: authentication.UserTokenPayload) -> None:
        # Stores tokens in .tio.tokens.json by default; can be overriden to use a DB for example
        # Adds the token to our Client to make requests and subscribe to EventSub...
        await self.add_token(payload.access_token, payload.refresh_token)

        if payload.user_id == self.bot_id:
            return
        # Subscribe to chat for new authorizations...
        chat = eventsub.ChatMessageSubscription(broadcaster_user_id=payload.user_id, user_id=self.bot_id)
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
        This is the core of the message fetching for Twitch.
        """
        # Transform the message into UnifiedMessage format
        unified_msg = UnifiedMessage(
            platform='Twitch',
            username=payload.chatter.name,
            content=payload.text,
            timestamp=payload.timestamp, # Use the notification timestamp
            color=_hex_to_rgb(str(payload.color))
        )
        
        # Append the message to the shared list
        self.bot.message_list.append(unified_msg)
        logger.info(f"Twitch message captured from {unified_msg.username}: {unified_msg.content}")

async def fetch_twitch_user_ids(client_id: str, client_secret: str, channel_name: str, bot_name: str) -> Tuple[str, str]:
    """
    Fetches and returns user IDs for the channel owner and the bot using App credentials.
    This is a pure utility function without side effects.
    """
    logger.info(f"Fetching Twitch user IDs for channel '{channel_name}' and bot '{bot_name}'...")
    try:
        async with twitchio.Client(client_id=client_id, client_secret=client_secret) as client:
            await client.login()
            users = await client.fetch_users(logins=[channel_name, bot_name])
            if len(users) < 2:
                raise ValueError("Could not fetch both channel and bot users. Check names.")
            
            # Assuming the first name in the list corresponds to the first user found
            name_map = {user.name.lower(): user.id for user in users}
            owner_id = name_map.get(channel_name.lower())
            bot_id = name_map.get(bot_name.lower())

            if not owner_id or not bot_id:
                raise ValueError(f"Could not resolve both user IDs from response. Got: {name_map}")

            logger.info(f"Fetched IDs: Owner={owner_id}, Bot={bot_id}")
            return owner_id, bot_id
    except Exception as e:
        logger.critical(f"Could not fetch owner/bot IDs. Please check credentials and names. Error: {e}")
        raise


### The section below is for standalone testing.

async def _test_twitch_bot():
    """Main function to run the bot as a standalone application for testing."""
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(filename)s - %(funcName)s] %(message)s")
    logger.info("--- Running Twitch Bot Standalone Test ---")
    
    from dotenv import load_dotenv
    from general_utils import get_env_var
    load_dotenv()

    try:
        # Load all configuration from environment for the test
        client_id = get_env_var("TW_CLIENT_ID", str)
        client_secret = get_env_var("TW_CLIENT_SECRET", str)
        channel_name = get_env_var("TW_CHANNEL", str)
        bot_name = get_env_var("TW_BOT_NAME", str)
        
        # In a real app, these would be fetched once and stored persistently.
        owner_id, bot_id = await fetch_twitch_user_ids(client_id, client_secret, channel_name, bot_name)
        
        test_message_list: Deque[UnifiedMessage] = Deque(maxlen=100)
        
        bot_instance = Bot(
            message_list=test_message_list,
            client_id=client_id,
            client_secret=client_secret,
            bot_id=bot_id,
            owner_id=owner_id,
            prefix="!",
        )
        
        async def print_messages():
            while True:
                if test_message_list:
                    msg = test_message_list.popleft()
                    logger.info(f"  -> [TEST] Message received and captured: {msg}")
                await asyncio.sleep(1)

        await asyncio.gather(bot_instance.start(), print_messages())

    except Exception as e:
        logger.error(f"An error occurred during standalone test: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(_test_twitch_bot())
    except KeyboardInterrupt:
        logger.warning("Shutting down test due to KeyboardInterrupt.")