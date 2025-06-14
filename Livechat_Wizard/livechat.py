import threading
from general_utils import get_env_var
from youtube import YTLive
from twitch import TwitchAuth, Bot
from kick import KickClient
from livechat_utils import ChatPicker
import time
import multiprocessing

class LiveChatController:
    def __init__(self, fetch_youtube=False, fetch_twitch=False, fetch_kick=False):
        self.youtube = None
        self.next_page_token = get_env_var("LAST_NEXT_PAGE_TOKEN") 
        self.twitch_bot = None
        self.KICK_CLIENT = None
        chat_sources = []

        if fetch_youtube:
            self.yt_messages = []
            chat_sources.append(self.yt_messages)
            self.setup_youtube()
        
        if fetch_twitch:
            manager = multiprocessing.Manager()
            self.twitch_chat_msgs = manager.list()
            chat_sources.append(self.twitch_chat_msgs)
            self.setup_twitch()

        if fetch_kick:
            self.kick_messages = []  # Placeholder for Kick messages
            chat_sources.append(self.kick_messages)
            self.setup_kick()

        #only desired chats should be included in the random message picking
        self.picker = ChatPicker(*chat_sources)

    @classmethod
    def create(cls):
        fetch_youtube = get_env_var("YT_FETCH")
        fetch_twitch = get_env_var("TW_FETCH")
        fetch_kick = get_env_var("KI_FETCH")

        # Return None if all fetch variables are False
        if not any([fetch_youtube, fetch_twitch, fetch_kick]):
            return None

        return cls(fetch_youtube=fetch_youtube, fetch_twitch=fetch_twitch, fetch_kick=fetch_kick)

    #get token and prepare for fetching youtube livechat messages
    def setup_youtube(self):
        self.youtube = YTLive(self.yt_messages)

    #get token and start twitch bot on a separate thread for livechat messages
    @staticmethod
    def _twitch_process(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN, twitch_chat_msgs):
        twitch_bot = Bot(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN, twitch_chat_msgs)
        twitch_bot.run()

    def setup_twitch(self):
        TW_Auth = TwitchAuth()
        CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, USE_THIRD_PARTY_TOKEN = TW_Auth.CHANNEL, TW_Auth.BOT_NICK, TW_Auth.CLIENT_ID, TW_Auth.CLIENT_SECRET, TW_Auth.ACCESS_TOKEN, TW_Auth.USE_THIRD_PARTY_TOKEN
        if USE_THIRD_PARTY_TOKEN:
            TOKEN = ACCESS_TOKEN
        elif not ACCESS_TOKEN:
            TOKEN = TW_Auth.access_token_generator()
        else:
            TOKEN = TW_Auth.refresh_access_token()
        twitch_bot_process = multiprocessing.Process(target=self._twitch_process, args=(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN, self.twitch_chat_msgs), daemon=True)
        twitch_bot_process.start()
    
    #WIP
    def setup_kick(self):
        kick_channel = get_env_var("KI_CHANNEL")
        self.KICK_CLIENT = KickClient(username=kick_channel, kick_chat_msgs=self.kick_messages)
        # KICK_CLIENT.listen() #uncomment for retrieving messages as they come in*

    #fetch a random message from 
    async def fetch_chat_message(self):
        #fetch raw youtube messages and process them automatically -- adds automatically to yt_messages
        if self.youtube:
            self.next_page_token = self.youtube.get_live_chat_messages(next_page_token=self.next_page_token)
        
        #fetch raw kick messages and process them automatically -- adds automatically to kick_messages
        if self.KICK_CLIENT:
            raw_messages = self.KICK_CLIENT.fetch_raw_messages(num_to_fetch=10)
            self.KICK_CLIENT.process_messages(raw_messages)

        message = self.picker.pick_rand_message()
        print("PICKED MESSAGE:", message)
        return message


# Example usage:
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    load_dotenv()

    fetch_youtube = get_env_var("YT_FETCH") 
    fetch_twitch = get_env_var("TW_FETCH")
    fetch_kick = get_env_var("KI_FETCH")
    live_chat_setup = LiveChatController.create()#(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube, fetch_kick=fetch_kick)

    while True:
        print("attempting_fetch")
        asyncio.run(live_chat_setup.fetch_chat_message())
        time.sleep(5.5)  # Adjust the interval as needed
