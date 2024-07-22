# from youtube import YTLive, YTTools
# from livechat_utils import ChatPicker 
# from twitch import TwitchAuth, TwitchTools, Bot, twitch_chat_msgs
# import threading

# class Livechats():
#     def __init__(self, Youtube=None, Twitch=None, Kick=None):
#         self.Youtube = Youtube
#         self.Twitch = Twitch
#         self.Kick = Kick
        
#     def setup_twitch_bot():
#         CHANNEL = 'scikous'
#         BOT_NICK = 'Botty'

#         TWTools = TwitchTools()
#         CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
#         TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)

#         # # Replace with your Twitch token and channel
#         TOKEN = TW_Auth.auth_access_token()
#         bot = Bot(TOKEN,CLIENT_ID, BOT_NICK, CHANNEL)
#         bot.run()
# #basic setup to start fetching live chat messages from live stream
# def youtube_setup(video_id):
#     api_key = YTTools.api_key_loader('livechatAPI/credentials/youtube.json')
#     live_chat_id = YTTools.get_live_chat_id(video_id=video_id)

#     YTLIVE = YTLive(api_key=api_key, live_chat_id=live_chat_id)
#     return YTLIVE



# async def livechats_setup(fetch_youtube=False, fetch_twitch=False, fetch_kick=False):
#     if fetch_youtube:
#         YTLIVE = youtube_setup(video_id="oQYdDy86eoY")
#     if fetch_twitch:
#         twitch_thread = threading.Thread(target=run_twitch_bot, daemon=True)
#         twitch_thread.start()
#     if fetch_kick:
#         pass
#     # print(twitch_chat_msgs)
    
# def fetch_chat_msgs()
#     yt_messages, next_page_token = YTLIVE.get_live_chat_messages()
#     kick = []
#     picker = ChatPicker(yt_messages, twitch_chat_msgs, kick)

#     message = picker.pick_rand_message()
#     print(message)
#     if yt_messages:
#         return yt_messages[0][1], next_page_token
#     else:
#         return None, None

# # import asyncio
# # msg = asyncio.run(fetch_chat_msgs())
# # print(msg[0])



# ##################################
# # {
# #     "yt": (
# #         "user": "msg",
# #         "user2": "msg2",
# #     )
# #     "tw": (
# #         {
# #             "user": "msg",
# #             "user2": "msg2",
# #         }
# #     )
# #     "ki": (
# #         "user": "msg",
# #         "user2": "msg2",
# #     )
# # }

# # {
# #     "yt": (
# #         (user, msg),
# #         (user2, msg2)
# #     ),
# #     "tw": (
# #         (user, msg),
# #         (user2, msg2)
# #     ),
# #     "ki": (
# #         (user, msg),
# #         (user2, msg2)
# #     )
# # }

# #########
#     #     csv_reader = csv.reader(file)
#     #     messages = tuple((tuple(row) for row in csv_reader))
#     # return messages[-num_messages:]


# # file = 'livechatAPI/data/kick_chat.csv'
# # # messages_to_csv(file, (("john", "msg0"), ("bob","msg")))
# # messages = read_messages_csv(file)
# # print(messages)


# # ############################
# # from twitch import TwitchTools, TwitchAuth, Bot
# # import threading
# # TWTools = TwitchTools()
# # CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
# # CHANNEL = 'scikous'
# # BOT_NICK = 'Botty'

# # TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)


# # # Replace with your Twitch token and channel
# # TOKEN = TW_Auth.auth_access_token()
# # bot = Bot(TOKEN,CLIENT_ID, BOT_NICK, CHANNEL)
# # twitch_thread = threading.Thread(target=bot.run, daemon=True)
# # twitch_thread.start()






# import threading
# from youtube import YTLive, YTTools
# from livechat_utils import ChatPicker 
# from twitch import TwitchAuth, TwitchTools, Bot, twitch_chat_msgs
# import time



# def setup_twitch():
#     CHANNEL = 'scikous'
#     BOT_NICK = 'Botty'

#     TWTools = TwitchTools()
#     CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
#     TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)

#     # Replace with your Twitch token and channel
#     TOKEN = TW_Auth.auth_access_token()
#     bot = Bot(TOKEN, CLIENT_ID, BOT_NICK, CHANNEL)
#     return bot



# def setup_youtube():
#     api_key = YTTools.api_key_loader('livechatAPI/credentials/youtube.json')
#     youtube = YTLive(api_key=api_key, video_id="oQYdDy86eoY")
#     return youtube

# bot = setup_twitch()
# twitch_thread = threading.Thread(target=bot.run, daemon=True)
# twitch_thread.start()
# youtube = setup_youtube()

# def fetch_chat_msgs():
#     yt_messages = []
#     if youtube:
#         yt_messages, next_page_token = youtube.get_live_chat_messages(next_page_token)
#     else:
#         next_page_token = None
    
#     kick = []  # Placeholder for Kick messages
#     picker = ChatPicker(yt_messages, twitch_chat_msgs, kick)

#     message = picker.pick_rand_message()
#     print(message)
#     return message
# while True:
#     fetch_chat_msgs()
#     time.sleep(20)

# # ##kick api
# from kick_chat import client

# p = client.Client(username="scikous")
# print(p.listen())

# kick_thread= threading.Thread(target=p.listen, daemon=True)
# kick_thread.start()



















import threading
from youtube import YTLive, yt_messages
from livechat_utils import ChatPicker 
from twitch import TwitchAuth, TwitchTools, Bot, twitch_chat_msgs
import time

kick = []  # Placeholder for Kick messages
class LiveChatSetup:
    def __init__(self, fetch_twitch=False, fetch_youtube=False):
        self.fetch_twitch = fetch_twitch
        self.fetch_youtube = fetch_youtube
        self.twitch_bot = None
        self.youtube = None
        self.next_page_token = None
        self.picker = ChatPicker(yt_messages, twitch_chat_msgs, kick)

        if self.fetch_twitch:
            self.setup_twitch()

        if self.fetch_youtube:
            self.setup_youtube()

    def setup_twitch(self):
        CHANNEL = 'scikous'
        BOT_NICK = 'Botty'

        TWTools = TwitchTools()
        CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
        TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)

        # Replace with your Twitch token and channel
        TOKEN = TW_Auth.auth_access_token()
        self.twitch_bot = Bot(TOKEN, CLIENT_ID, BOT_NICK, CHANNEL)
        twitch_thread = threading.Thread(target=self.twitch_bot.run, daemon=True)
        twitch_thread.start()

    def setup_youtube(self):
        youtube_creds_file = 'livechatAPI/credentials/youtube.json'
        self.youtube = YTLive(youtube_creds_file)

    async def fetch_chat_message(self):
        if self.youtube:
            self.next_page_token = self.youtube.get_live_chat_messages(next_page_token=self.next_page_token)
        message = self.picker.pick_rand_message()
        # print(message)
        return message

# Example usage:
if __name__ == "__main__":
    import asyncio
    fetch_twitch = False
    fetch_youtube = True

    live_chat_setup = LiveChatSetup(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube)

    while True:
        asyncio.run(live_chat_setup.fetch_chat_message())
        time.sleep(10)  # Adjust the interval as needed
