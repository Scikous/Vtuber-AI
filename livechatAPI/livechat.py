import threading
from youtube import YTLive, yt_messages
from livechat_utils import ChatPicker 
from twitch import TwitchAuth, Bot, twitch_chat_msgs
import time

kick = []  # Placeholder for Kick messages

#
class LiveChatController:
    def __init__(self, fetch_twitch=False, fetch_youtube=False, fetch_kick=False):
        self.twitch_bot = None
        self.youtube = None
        self.next_page_token = None
        chat_sources = []

        if fetch_twitch:
            chat_sources.append(twitch_chat_msgs)
            self.setup_twitch()

        if fetch_youtube:
            chat_sources.append(yt_messages)
            self.setup_youtube()
        
        if fetch_kick:
            chat_sources.append(kick)
            self.setup_kick()

        #only desired chats should be included in the random message picking
        self.picker = ChatPicker(*chat_sources)

    #get token and start twitch bot on a separate thread for livechat messages
    def setup_twitch(self):
        TW_Auth = TwitchAuth()
        CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, THIRD_PARTY_TOKEN = TW_Auth.CHANNEL, TW_Auth.BOT_NICK, TW_Auth.CLIENT_ID, TW_Auth.CLIENT_SECRET, TW_Auth.ACCESS_TOKEN, TW_Auth.THIRD_PARTY_TOKEN
        if THIRD_PARTY_TOKEN:
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

    #get token and prepare for fetching youtube livechat messages
    def setup_youtube(self):
        self.youtube = YTLive()

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
    fetch_twitch = True
    fetch_youtube = False

    live_chat_setup = LiveChatController(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube)
    while True:
        asyncio.run(live_chat_setup.fetch_chat_message())
        time.sleep(15.5)  # Adjust the interval as needed
