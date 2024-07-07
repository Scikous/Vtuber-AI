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



###################################################
####get oauth token WIP
from requests_oauthlib import OAuth2Session
import webbrowser

def twitch_auth_loader(cred_file):
    with open(cred_file, 'r') as credentials:
        creds = json.load(credentials)
        CLIENT_ID = creds["client-id"]
        CLIENT_SECRET = creds["client-secret"]
    return CLIENT_ID, CLIENT_SECRET


CLIENT_ID, CLIENT_SECRET = twitch_auth_loader("livechatAPI/credentials/twitch.json")
def twit():
    REDIRECT_URI = 'https://localhost:8080'
    AUTHORIZATION_BASE_URL = 'https://id.twitch.tv/oauth2/authorize'
    TOKEN_URL = 'https://id.twitch.tv/oauth2/token'

    try:
        # Create an OAuth2 session object
        oauth = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["chat:read", "chat:edit"])

        # Get the authorization URL and state parameter
        authorization_url, state = oauth.authorization_url(AUTHORIZATION_BASE_URL)

        # Open the authorization URL in the browser
        webbrowser.open(authorization_url)
        print(authorization_url)
        # print(f'Please go to {authorization_url} and authorize access.')
        # Get the full redirect URL after authorization
        redirect_response = input('Paste the full redirect URL here: ')
        # Fetch the access token
        token = oauth.fetch_token(TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET,include_client_id=True, authorization_response=redirect_response)
        # print('Access Token:', token)
        print('\n'*25+'#'*30)
        return token["access_token"]
    except Exception as e:
        print(f'HTTPError: {e}')
    # except Exception as e:
    #     print(f'Error: {e}')
    #####

token = twit()

# from twitchio.ext import commands
# # # Replace with your Twitch token and channel
# # TOKEN = token
# # CHANNEL = 'scikous'
# # BOT_NICK = 'Botty'
# # class Bot(commands.Bot):

# #     def __init__(self):
# #         super().__init__(token=TOKEN, client_id=CLIENT_ID, nick=BOT_NICK, prefix='!', initial_channels=[CHANNEL])

# #     async def event_ready(self):
# #         print(f'Ready | {self.nick}')

# #     async def event_message(self, message):
# #         print(f'{message.author.name}: {message.content}')
# #         await self.handle_commands(message)

# #     @commands.command(name='hello')
# #     async def hello(self, ctx):
# #         await ctx.send(f'Hello {ctx.author.name}!')

# # if __name__ == "__main__":
# #     bot = Bot()
# #     bot.run()
# #############################


# # ##kick api
# # from kick_chat import client

# # p = client.Client(username="username")
# # print(p.listen())

