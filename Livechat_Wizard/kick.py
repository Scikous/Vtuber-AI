# import sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
from datetime import datetime, timezone
from curl_cffi import requests
from general_utils import get_env_var
from livechat_utils import append_livechat_message
import dotenv

def hex_to_rgb(val: str) -> tuple[int, int, int]:
    return (int(val[i : i + 2], 16) for i in (1, 3, 5))

class KickClient:
    def __init__(self, *, username: str, kick_chat_msgs):
        self.username = username
        self.kick_chat_msgs = kick_chat_msgs
        self.channel_id = self.get_channel_id()
        #either a defined last timestamp or defaults to min UTC datetime
        self.last_message_time = get_env_var("KI_LAST_FETCH_TIME") or datetime.min.replace(tzinfo=timezone.utc)

    #kick channel ID for fetching live chat messages from
    def get_channel_id(self):
        res = requests.get(
            f"https://kick.com/api/v2/channels/{self.username}",
            impersonate="chrome110",
        )
        res.raise_for_status()
        data = res.json()
        return data["id"]

    #returns all messages and their data, also no need to fetch all
    def fetch_raw_messages(self, num_to_fetch=10):
        """
        Fetches all messages and their respective data (timestamp, user, message, color, etc.)

        num_to_fetch (int): -1 fetches all messages  
        """
        url = f"https://kick.com/api/v2/channels/{self.channel_id}/messages"
        res = requests.get(url, impersonate="chrome110")
        res.raise_for_status()
        data = res.json().get("data", [])
        messages_data = data["messages"]
        if num_to_fetch == -1:
            return messages_data
        return data["messages"][:num_to_fetch]

    #used alongside listen, writes message to console, appends to running list
    def process_message(self, message):
        """
        Used when using listen() method

        Write message to console and append to running list
        """
        message_time = datetime.fromisoformat(message["created_at"].replace("Z", "+00:00"))
        time_str = message_time.strftime("%H:%M:%S")
        content = message["content"]
        user = message["sender"]["username"]
        
        r, g, b = hex_to_rgb(message["sender"]["identity"]["color"])
        fgColorString = f"\033[38;2;{r};{g};{b}m"
        print(f"[{time_str}] {fgColorString}{user}\033[0m: {content}")
        user_msg = (user, content)
        self.kick_chat_msgs.append(user_msg)
        self.last_message_time = max(self.last_message_time, message_time)
        dotenv.set_key(dotenv_path=dotenv.find_dotenv(),key_to_set="KI_LAST_FETCH_TIME", value_to_set=message["created_at"])
        print(user_msg)

    #basically intented for polling -- fetch_raw_messages -> process_messages
    def process_messages(self, messages):
        """
        Used as a standalone

        append messages to running list
        """
        if messages:
            #get only username and message -- (<username>, <message>)
            new_clean_messages = [
                (msg["sender"]["username"], msg['content']) for msg in messages 
                if datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")) > self.last_message_time
            ]

            #add newest messages to kick live chat list
            for user_message in new_clean_messages:
                append_livechat_message(self.kick_chat_msgs, user_message)
            
            #most recent message is always first, set last fetch time to it
            recent_message_time = messages[0]["created_at"]
            dotenv.set_key(dotenv_path=dotenv.find_dotenv(),key_to_set="KI_LAST_FETCH_TIME", value_to_set=recent_message_time)
            self.last_message_time = datetime.fromisoformat(recent_message_time.replace("Z", "+00:00"))
        else:
            raise ValueError("No messages were provided!")
    
    #technically not constantly listening, but mimics it all the same
    def listen(self):
        """
        "listens" for new messages -- technicallly fetches periodically
        """
        print(f"Fetching messages for channel: {self.username}")
        while True:
            messages = self.fetch_raw_messages()
            if not messages:
                print("No messages received. Waiting before trying again...")
                time.sleep(5)
                continue

            new_messages = [
                msg for msg in messages 
                if datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")) > self.last_message_time
            ]
            
            for message in reversed(new_messages):
                self.process_message(message)
            
            time.sleep(5)  # Wait for 5 seconds before fetching again

# Usage
if __name__ == "__main__":
    from general_utils import get_env_var
    username = get_env_var("KI_CHANNEL")
    kick_chat_msgs = []
    client = KickClient(username=username, kick_chat_msgs=kick_chat_msgs)
    client.listen()
    messages = client.fetch_raw_messages()
    client.process_messages(messages)
    print(client.kick_chat_msgs)