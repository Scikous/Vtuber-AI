"""
Environment Variable Utility for Vtuber-AI
Provides functions for fetching environment variables.
"""
import dotenv
from dateutil.parser import parse

def get_env_var(env_var):
    """
    Fetches information from .env file.

    Args:
        env_var (str): Any environment variable in .env file.
    Returns:
        bool | str | int | float | datetime | None: The value of the environment variable, converted to an appropriate type, or None if not found.
    """
    # Ensure .env is loaded if not already by the main application
    # dotenv.load_dotenv() # Typically called once at application start

    env_key = dotenv.get_key(dotenv_path=dotenv.find_dotenv(), key_to_get=env_var)

    if env_key is None:
        return None

    try:
        return int(env_key)
    except ValueError:
        try:
            return float(env_key)
        except ValueError:
            # Attempt to parse as datetime
            try:
                return parse(env_key)
            except ValueError:
                if env_key.lower() == "true":
                    return True
                elif env_key.lower() == "false":
                    return False
                else:
                    return env_key