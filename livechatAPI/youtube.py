from googleapiclient.discovery import build
from general_utils import get_env_var
import dotenv

class YTLive():
    def __init__(self, yt_messages):
        API_KEY, LIVESTREAM_ID = self.youtube_credentials()
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)
        self.live_chat_id = self.get_live_chat_id(LIVESTREAM_ID)
        self.yt_messages = yt_messages
        # request = self.youtube.liveStreams().list(
        #     part='id,snippet',
        #     mine=True  # Filter for your own live streams
        # )
        # response = request.execute()
        # print(response)

    #retrieves the API key from .env
    @staticmethod
    def youtube_credentials():
        API_KEY, LIVESTREAM_ID = get_env_var("YT_API_KEY"), get_env_var("LIVESTREAM_ID")
        return API_KEY, LIVESTREAM_ID
    
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
            self.yt_messages.extend(yt_new_messages)
            next_page_token = response.get('nextPageToken')
            #save current next_page_token to ENV variable
            dotenv.set_key(dotenv_path=dotenv.find_dotenv(),key_to_set="LAST_NEXT_PAGE_TOKEN",value_to_set=next_page_token)
        return next_page_token
    
if __name__ == "__main__":
    yt_messages = []
    YTLive(yt_messages)