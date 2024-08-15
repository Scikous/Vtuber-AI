from collections import deque
import csv
import dotenv

#save user+LLM message(s) for convenient data
async def write_messages_csv(file_path, message_data):
    '''
    Intented for writing tuples of chat message_data -> ('<user name>: <message>', '<LLM output>')

    Can technically be any format
    '''
    # global lock #we may come back to threading eventually
    with open(file_path, mode='a', newline='\n', encoding='utf-8') as file:
        # csv_writer = csv.writer(file)
        # # with lock:
        # csv_writer.writerow(message_data)
        csv_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(message_data)


#fetch last 10 messages from csv
def read_messages_csv(file_path, num_messages=10):
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
        return int(env_key)
    except ValueError:
        try:
            return float(env_key)
        except ValueError:
            if env_key.lower() == "true":
                return True
            elif env_key.lower() == "false":
                return False
            else:
                return env_key