import json
from googleapiclient.discovery import build
from livechat_utils import write_messages_csv

yt_messages = []
DEFAULT_SAVE_FILE = "livechatAPI/data/youtube_chat.csv"
    
class YTLive():
    def __init__(self, youtube_creds_file, SAVE_MESSAGES=True):
        api_key, video_id = self.youtube_credentials(youtube_creds_file)
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.live_chat_id = self.get_live_chat_id(video_id)
        self.write_message_func = write_messages_csv if SAVE_MESSAGES else None  # Assign conditionally

    @staticmethod
    def youtube_credentials(cred_file):
        with open(cred_file, 'r') as credentials:
            creds = json.load(credentials)
            api_key, video_id = creds["api_key"], creds["video_id"]
        return api_key, video_id
    
    def get_live_chat_id(self, video_id):
        response = self.youtube.videos().list(
            part='liveStreamingDetails',
            id=video_id
        ).execute()
        live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
        return live_chat_id
    
    #fetch youtube livestream chat messages
    def get_live_chat_messages(self, next_page_token=None):
        response = self.youtube.liveChatMessages().list(
            liveChatId=self.live_chat_id,
            part='snippet,authorDetails',
            pageToken=next_page_token
        ).execute()
        messages = response.get('items', [])
        if messages:
            yt_new_messages = [(message['authorDetails']['displayName'],message['snippet']['displayMessage']) for message in messages]
            #write message to file -- stability is questionable for bigger stream chats
            if self.write_message_func:
                self.write_message_func(DEFAULT_SAVE_FILE, yt_new_messages)
            yt_messages.extend(yt_new_messages)
            next_page_token = response.get('nextPageToken')
        return next_page_token