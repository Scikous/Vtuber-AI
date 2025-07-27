from collections import deque
import csv
import dotenv
from dateutil.parser import parse
from functools import wraps
from contextlib import contextmanager
import time
import os


#save user+LLM message(s) for convenient data
async def write_messages_csv(file_path, message_data):
    '''
    Intented for writing tuples of chat message_data -> ('<user name>: <message>', '<LLM output>')

    Can technically be any format
    '''
    # global lock #we may come back to threading eventually
    with open(file_path, mode='a+', newline='\n', encoding='utf-8') as file:
        # csv_writer = csv.writer(file)
        # # with lock:
        # csv_writer.writerow(message_data)
        csv_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(message_data)


#fetch last 10 messages from csv
async def read_messages_csv(file_path, num_messages=10):
    '''
    Intented for return tuples of chat messages -> (<user name>, <message>)

    Can technically be any format

    By default returns the latest 10 chat messages as a list of tuples -> [(<user name>, <message>), ...]
    '''

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        messages = deque(csv_reader, maxlen=num_messages)
    return [tuple(row) for row in messages]

#get ENV variables for livechat API related needs
def get_env_var(env_var):
    """
    Fetches information from .env file.

    Args:
        env_var (str): Any environment variable in .env file.
    Returns:
        bool | str: True/False if the key is a boolean type, otherwise returns the key value itself.
    """
    env_key = dotenv.get_key(dotenv_path=dotenv.find_dotenv(), key_to_get=env_var)

    try:
        if not env_key:
            return None
        return int(env_key)
    except (TypeError, ValueError):
        try:
            return float(env_key)
        except (TypeError, ValueError):
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
            
@contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)

#used as a decorator for functions that can fail initially, namely YouTube/Twitch/Kick livechat
def retry_with_backoff(max_retries=3, initial_delay=5, backoff_factor=2, exceptions=(Exception,)):
    """
    A decorator that retries the decorated function with exponential backoff.

    :param max_retries: Maximum number of retries before giving up
    :param initial_delay: Initial delay between retries in seconds
    :param backoff_factor: Multiplier for delay after each retry
    :param exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed with the following ERROR: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator