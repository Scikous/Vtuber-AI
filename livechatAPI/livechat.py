import threading
from youtube import YTLive, yt_messages
from livechat_utils import ChatPicker, get_env_var
from twitch import TwitchAuth, Bot, twitch_chat_msgs
import time

class LiveChatController:
    def __init__(self, fetch_youtube=False, fetch_twitch=False, fetch_kick=False):
        self.twitch_bot = None
        self.youtube = None
        self.next_page_token = None
        chat_sources = []

        if fetch_youtube:
            chat_sources.append(yt_messages)
            self.setup_youtube()
        
        if fetch_twitch:
            chat_sources.append(twitch_chat_msgs)
            self.setup_twitch()

        if fetch_kick:
            kick = []  # Placeholder for Kick messages
            chat_sources.append(kick)
            self.setup_kick()

        #only desired chats should be included in the random message picking
        self.picker = ChatPicker(*chat_sources)

    #get token and prepare for fetching youtube livechat messages
    def setup_youtube(self):
        self.youtube = YTLive()

    #get token and start twitch bot on a separate thread for livechat messages
    def setup_twitch(self):
        TW_Auth = TwitchAuth()
        CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, USE_THIRD_PARTY_TOKEN = TW_Auth.CHANNEL, TW_Auth.BOT_NICK, TW_Auth.CLIENT_ID, TW_Auth.CLIENT_SECRET, TW_Auth.ACCESS_TOKEN, TW_Auth.USE_THIRD_PARTY_TOKEN
        if USE_THIRD_PARTY_TOKEN:
            TOKEN = ACCESS_TOKEN
        elif not ACCESS_TOKEN:
            TOKEN = TW_Auth.access_token_generator()
        else:
            TOKEN = TW_Auth.refresh_access_token()
        self.twitch_bot = Bot(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN)
        twitch_thread = threading.Thread(target=self.twitch_bot.run, daemon=True)
        twitch_thread.start()
    
    #WIP
    def setup_kick(self):
        pass


    #fetch a random message from 
    async def fetch_chat_message(self):
        if self.youtube:
            self.next_page_token = self.youtube.get_live_chat_messages(next_page_token=self.next_page_token)
        message = self.picker.pick_rand_message()
        # print("PICKED MESSAGE:", message)
        return message


# Example usage:
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()


    fetch_youtube = get_env_var("YT_FETCH") 
    fetch_twitch = get_env_var("TW_FETCH")
    fetch_kick = get_env_var("KI_FETCH")
    live_chat_setup = LiveChatController(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube, fetch_kick=fetch_kick)

    while True:
        asyncio.run(live_chat_setup.fetch_chat_message())
        time.sleep(15.5)  # Adjust the interval as needed
