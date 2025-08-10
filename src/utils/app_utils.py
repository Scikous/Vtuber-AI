# Livechat_Wizard/config_builder.py
import logging
from src.utils.env_utils import get_env_var
from Livechat_Wizard.twitch import fetch_twitch_user_ids

logger = logging.getLogger(__name__)

async def build_livechat_controller_config() -> dict:
    """
    Loads all necessary settings from the environment, prepares them,
    and returns a configuration dictionary suitable for LiveChatController.

    This function acts as the bridge between the environment/dotenv and the
    decoupled controller. It also handles first-time setup logic like
    fetching Twitch IDs.
    """
    logger.info("Building LiveChatController configuration...")

    # Base configuration structure
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
            "bot_id": get_env_var("TW_BOT_ID", str),
            "owner_id": get_env_var("TW_OWNER_ID", str),
            "prefix": "!",
            "max_messages": get_env_var("TWITCH_MAX_MESSAGES", int, 100)
        },
        "kick": {
            "enabled": get_env_var("KI_FETCH", bool, False),
            "channel_name": get_env_var("KI_CHANNEL", str)
        }
    }

    # Handle dynamic/first-run configuration for Twitch
    twitch_config = config["twitch"]
    if twitch_config["enabled"] and (not twitch_config["bot_id"] or not twitch_config["owner_id"]):
        logger.warning("Twitch Bot ID or Owner ID not found in environment. Attempting to fetch them...")
        try:
            owner_id, bot_id = await fetch_twitch_user_ids(
                client_id=twitch_config["client_id"],
                client_secret=twitch_config["client_secret"],
                channel_name=twitch_config["channel_name"],
                bot_name=twitch_config["bot_name"]
            )
            # Update the config dictionary with the fetched values
            twitch_config["owner_id"] = owner_id
            twitch_config["bot_id"] = bot_id
            logger.info(f"Successfully fetched Twitch IDs. Please add TW_OWNER_ID={owner_id} and TW_BOT_ID={bot_id} to your .env file to speed up future launches.")
        except Exception as e:
            logger.error(f"FATAL: Could not fetch Twitch IDs. Twitch support will be disabled. Error: {e}")
            twitch_config["enabled"] = False

    if not any(c.get('enabled') for c in config.values()):
        logger.warning("All fetch services are disabled in the configuration. The controller will not fetch from any platform.")

    return config