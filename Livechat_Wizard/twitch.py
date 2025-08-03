from general_utils import get_env_var
import dotenv

# Load environment variables
dotenv.load_dotenv()
CHANNEL = get_env_var("TW_CHANNEL", var_type=str)
BOT_NAME = get_env_var("TW_BOT_NAME", var_type=str)
CLIENT_ID = get_env_var("TW_CLIENT_ID", var_type=str)
CLIENT_SECRET = get_env_var("TW_CLIENT_SECRET", var_type=str)
BOT_ID = get_env_var("TW_BOT_ID", var_type=str)
OWNER_ID = get_env_var("TW_OWNER_ID", var_type=str)

import asyncio
import json
import logging
import random
from typing import Any
import twitchio
from twitchio import authentication, eventsub
from twitchio.ext import commands


# NOTE: Consider reading through the Conduit examples
# Store and retrieve these from a .env or similar, but for example showcase you can just full out the below:

LOGGER: logging.Logger = logging.getLogger(__name__)


class Bot(commands.Bot):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    async def setup_hook(self) -> None:
        # Add our General Commands Component...
        await self.add_component(GeneralCommands())

        with open(".tio.tokens.json", "rb") as fp:
            tokens = json.load(fp)

        for user_id in tokens:
            if user_id == BOT_ID:
                continue

            # Subscribe to chat for everyone we have a token...
            chat = eventsub.ChatMessageSubscription(broadcaster_user_id=user_id, user_id=BOT_ID)
            await self.subscribe_websocket(chat)

    async def event_ready(self) -> None:
        LOGGER.info("Logged in as: %s", self.user)

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
    @commands.command()
    async def hi(self, ctx: commands.Context) -> None:
        """Command that replys to the invoker with Hi <name>!

        !hi
        """
        await ctx.reply(f"Hi {ctx.chatter}!")

    @commands.command()
    async def say(self, ctx: commands.Context, *, message: str) -> None:
        """Command which repeats what the invoker sends.

        !say <message>
        """
        await ctx.send(message)

    @commands.command()
    async def add(self, ctx: commands.Context, left: int, right: int) -> None:
        """Command which adds to integers together.

        !add <number> <number>
        """
        await ctx.reply(f"{left} + {right} = {left + right}")

    @commands.command()
    async def choice(self, ctx: commands.Context, *choices: str) -> None:
        """Command which takes in an arbitrary amount of choices and randomly chooses one.

        !choice <choice_1> <choice_2> <choice_3> ...
        """
        await ctx.reply(f"You provided {len(choices)} choices, I choose: {random.choice(choices)}")

    @commands.command(aliases=["thanks", "thank"])
    async def give(self, ctx: commands.Context, user: twitchio.User, amount: int, *, message: str | None = None) -> None:
        """A more advanced example of a command which has makes use of the powerful argument parsing, argument converters and
        aliases.

        The first argument will be attempted to be converted to a User.
        The second argument will be converted to an integer if possible.
        The third argument is optional and will consume the reast of the message.

        !give <@user|user_name> <number> [message]
        !thank <@user|user_name> <number> [message]
        !thanks <@user|user_name> <number> [message]
        """
        msg = f"with message: {message}" if message else ""
        await ctx.send(f"{ctx.chatter.mention} gave {amount} thanks to {user.mention} {msg}")



async def update_owner_bot_id() -> None:
    global OWNER_ID, BOT_ID
    async with twitchio.Client(client_id=CLIENT_ID, client_secret=CLIENT_SECRET) as client:
        await client.login()
        user = await client.fetch_users(logins=[CHANNEL, BOT_NAME])
        
        OWNER_ID = user[0].id
        dotenv.set_key(
            dotenv_path=dotenv.find_dotenv(),
            key_to_set="TW_OWNER_ID",
            value_to_set=OWNER_ID
        )

        BOT_ID = user[1].id
        dotenv.set_key(
                    dotenv_path=dotenv.find_dotenv(),
                    key_to_set="TW_BOT_ID",
                    value_to_set=BOT_ID
                )


def main() -> None:
    #use INFO when setting up, otherwise use CRITICAL to avoid accidental leaks
    twitchio.utils.setup_logging(level=logging.CRITICAL)

    async def runner() -> None:
        if not OWNER_ID or not BOT_ID:
            await update_owner_bot_id()
        async with Bot(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=BOT_ID,
            owner_id=OWNER_ID,
            prefix="!",
            logger=None
        ) as bot:
            await bot.start()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down due to KeyboardInterrupt")


if __name__ == "__main__":
    main()
