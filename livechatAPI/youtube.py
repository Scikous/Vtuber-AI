import json
import time
from googleapiclient.discovery import build




##wip

# Set your API key here
def api_key_loader(cred_file):
    with open(cred_file, 'r') as credentials:
        creds = json.load(credentials)
        api_key = creds["api_key"]
    return api_key

class YTLive():
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    def get_live_chat_id(self, video_id):
        response = self.youtube.videos().list(
            part='liveStreamingDetails',
            id=video_id
        ).execute()
        live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
        return live_chat_id

    def get_live_chat_messages(self, live_chat_id, pageToken=None):
        response = self.youtube.liveChatMessages().list(
            liveChatId=live_chat_id,
            part='snippet,authorDetails',
            pageToken=pageToken
        ).execute()
        messages = response.get('items', [])
        if messages:
            yt_messages = list((message['authorDetails']['displayName'],message['snippet']['displayMessage']) for message in messages)
            next_page_token = response.get('nextPageToken')
            return yt_messages, next_page_token
        return None, None
        # return messages
        # for message in messages:
        #     print(f"{message['authorDetails']['displayName']}: {message['snippet']['displayMessage']}")

    # Replace with your YouTube live video ID