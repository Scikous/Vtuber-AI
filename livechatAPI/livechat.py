import json
import time
from googleapiclient.discovery import build
################### youtube api working as expected
# def api_key_loader(cred_file):
#     with open(cred_file, 'r') as credentials:
#         creds = json.load(credentials)
#         api_key = creds["api_key"]
#     return api_key

# youtube_creds = api_key_loader('livechatAPI/credentials/youtube.json')


# ##wip

# # Set your API key here



# api_key = youtube_creds

# class YTLive():
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.youtube = build('youtube', 'v3', developerKey=api_key)

#     def get_live_chat_id(self, video_id):
#         response = self.youtube.videos().list(
#             part='liveStreamingDetails',
#             id=video_id
#         ).execute()
#         live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
#         return live_chat_id

#     def get_live_chat_messages(self, live_chat_id, pageToken=None):
#         response = self.youtube.liveChatMessages().list(
#             liveChatId=live_chat_id,
#             part='snippet,authorDetails',
#             pageToken=pageToken
#         ).execute()
#         messages = response.get('items', [])
#         if messages:
#             yt_messages = tuple((message['authorDetails']['displayName'],message['snippet']['displayMessage']) for message in messages )
#             next_page_token = response.get('nextPageToken')
#             return yt_messages, next_page_token
#         return None, None
#         # return messages
#         # for message in messages:
#         #     print(f"{message['authorDetails']['displayName']}: {message['snippet']['displayMessage']}")

#     # Replace with your YouTube live video ID

# video_id = ''
# YTLIVE = YTLive(api_key=youtube_creds) 
# live_chat_id = YTLIVE.get_live_chat_id(video_id)
# async def fetch_chat_msgs():
#     yt_messages, next_page_token = YTLIVE.get_live_chat_messages(live_chat_id)
#     if yt_messages:
#         return yt_messages[0][1], next_page_token
#     else:
#         return None, None
    


# import asyncio
# msg = asyncio.run(fetch_chat_msgs())
# print(msg[0])

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


############################
from twitch import TwitchTools, TwitchAuth, Bot
import threading
TWTools = TwitchTools()
CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
CHANNEL = 'scikous'
BOT_NICK = 'Botty'

TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)


# Replace with your Twitch token and channel
TOKEN = TW_Auth.auth_access_token()
# bot = Bot(TOKEN,CLIENT_ID, BOT_NICK, CHANNEL)
# twitch_thread = threading.Thread(target=bot.run, daemon=True)
# twitch_thread.start()

##kick api
from kick_chat import client

p = client.Client(username="scikous")
print(p.listen())