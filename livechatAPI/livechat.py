from youtube import YTLive, api_key_loader
from livechat_utils import ChatPicker, twitch_chat_msgs

youtube_creds = api_key_loader('livechatAPI/credentials/youtube.json')
api_key = youtube_creds

video_id = ''
YTLIVE = YTLive(api_key=youtube_creds) 
live_chat_id = YTLIVE.get_live_chat_id(video_id)

async def fetch_chat_msgs():
    yt_messages, next_page_token = YTLIVE.get_live_chat_messages(live_chat_id)
    print(twitch_chat_msgs)
    kick = []
    picker = ChatPicker(yt_messages, twitch_chat_msgs, kick)

    message = picker.pick_rand_message()
    print(message)
    if yt_messages:
        return yt_messages[0][1], next_page_token
    else:
        return None, None
    

import asyncio
msg = asyncio.run(fetch_chat_msgs())
print(msg[0])



##################################
# {
#     "yt": (
#         "user": "msg",
#         "user2": "msg2",
#     )
#     "tw": (
#         {
#             "user": "msg",
#             "user2": "msg2",
#         }
#     )
#     "ki": (
#         "user": "msg",
#         "user2": "msg2",
#     )
# }

# {
#     "yt": (
#         (user, msg),
#         (user2, msg2)
#     ),
#     "tw": (
#         (user, msg),
#         (user2, msg2)
#     ),
#     "ki": (
#         (user, msg),
#         (user2, msg2)
#     )
# }

#########
    #     csv_reader = csv.reader(file)
    #     messages = tuple((tuple(row) for row in csv_reader))
    # return messages[-num_messages:]


# file = 'livechatAPI/data/kick_chat.csv'
# # messages_to_csv(file, (("john", "msg0"), ("bob","msg")))
# messages = read_messages_csv(file)
# print(messages)


# ############################
# from twitch import TwitchTools, TwitchAuth, Bot
# import threading
# TWTools = TwitchTools()
# CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
# CHANNEL = 'scikous'
# BOT_NICK = 'Botty'

# TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)


# # Replace with your Twitch token and channel
# TOKEN = TW_Auth.auth_access_token()
# bot = Bot(TOKEN,CLIENT_ID, BOT_NICK, CHANNEL)
# twitch_thread = threading.Thread(target=bot.run, daemon=True)
# twitch_thread.start()

# # ##kick api
# from kick_chat import client

# p = client.Client(username="scikous")
# print(p.listen())