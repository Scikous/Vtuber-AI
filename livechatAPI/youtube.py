from googleapiclient.discovery import build
from general_utils import get_env_var, retry_with_backoff
from livechat_utils import append_livechat_message
import dotenv

class YTLive():
    def __init__(self, yt_messages):
        API_KEY, CHANNEL_ID = self.youtube_credentials()
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)
        self.livestream_id = self.get_livestream_id(CHANNEL_ID)
        self.live_chat_id = self.get_live_chat_id(livestream_id=self.livestream_id)
        self.yt_messages = yt_messages

    #retrieves the API key and ChannelID from .env
    @staticmethod
    def youtube_credentials():
        API_KEY, CHANNEL_ID = get_env_var("YT_API_KEY"), get_env_var("YT_CHANNEL_ID")
        return API_KEY, CHANNEL_ID
    
    #retrieves livestream ID -- used to fetch live chat id
    @retry_with_backoff(max_retries=5, initial_delay=5, backoff_factor=2, exceptions=(ValueError,))
    def get_livestream_id(self, channel_id):
        """
        channel_id: Found from YouTube.com -> Settings -> Advanced Settings
        """
        if channel_id:
            #get active livestream data
            response = self.youtube.search().list(
                part='id,snippet',
                channelId=channel_id,
                type='video',
                eventType='live',
                maxResults=1
            ).execute()
            if not response['items']:
                raise ValueError("No active livestreams found!")

            livestream_id = response['items'][0]['id']['videoId']
            return livestream_id
        else:
            raise ValueError("YouTube channel_id was not provided")
    
    #live chat id is used to retrieve live chat messages
    def get_live_chat_id(self, livestream_id=None):
        if livestream_id:
            #get active livestream's live chat id
            response = self.youtube.videos().list(
                part='liveStreamingDetails',
                id=livestream_id
            ).execute()

            live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
            return live_chat_id
        else:
            raise ValueError("No livestream ID provided!")

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
            append_livechat_message(self.yt_messages, yt_new_messages)
            next_page_token = response.get('nextPageToken')
            #save current next_page_token to ENV variable
            dotenv.set_key(dotenv_path=dotenv.find_dotenv(),key_to_set="LAST_NEXT_PAGE_TOKEN",value_to_set=next_page_token)
        return next_page_token
    
if __name__ == "__main__":
    yt_messages = []
    yt_live_controller = YTLive(yt_messages)
    yt_live_controller.get_live_chat_messages()
    print(len(yt_messages))
