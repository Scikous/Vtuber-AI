"""
Environment Variable Utility for Vtuber-AI
Provides functions for fetching environment variables.
"""
import dotenv
import sys, os
from dateutil.parser import parse

#get ENV variables for livechat API related needs
def get_env_var(env_var, var_type=str, default=None):
    """
    Fetches information from .env file.

    Args:
        env_var (str): Any environment variable in .env file.
    Returns:
        Uses type specified in var_type to convert the value of the environment variable, or None if not found.
    """
    env_key = dotenv.get_key(dotenv_path=dotenv.find_dotenv(), key_to_get=env_var)

    try:
        if not env_key:
            return default
        if var_type:
            return var_type(env_key)
    except TypeError:
        print(f"Error: {env_var} is not of type {var_type}")


def setup_project_root():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root