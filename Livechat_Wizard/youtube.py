### Uses OAuth2 to validate and access live chat
from googleapiclient.discovery import build
from general_utils import get_env_var, retry_with_backoff
import dotenv
import os.path
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class YTLive:
    def __init__(self, yt_messages: list):
        # Load channel ID from environment variable
        self.channel_id = get_env_var("YT_CHANNEL_ID")
        # Load OAuth2 credentials and build the YouTube service
        creds = self._load_credentials()
        self.youtube = build('youtube', 'v3', credentials=creds)
        # Get livestream and live chat IDs
        self.livestream_id = self.get_livestream_id()
        self.live_chat_id = self.get_live_chat_id()
        self.yt_messages = yt_messages

    def _load_credentials(self):
        # Define the scope for read-only access to YouTube live chat data
        SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
        creds = None
        # Check if token.pickle exists and load credentials
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # Validate credentials; refresh or obtain new ones if necessary
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                client_secret_YT = get_env_var("YT_OAUTH2_JSON")
                flow = InstalledAppFlow.from_client_secrets_file(
                   client_secret_YT, SCOPES)
                creds = flow.run_local_server(port=0)
                # Save new credentials to token.pickle
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
        return creds

    @retry_with_backoff(max_retries=7, initial_delay=5, backoff_factor=2, exceptions=(ValueError,))
    def get_livestream_id(self):
        """Retrieve the ID of an active livestream on the specified channel."""
        if self.channel_id:
            response = self.youtube.search().list(
                channelId=self.channel_id,
                part="snippet",
                eventType="live",  # Explicitly search for live events
                maxResults=1,     # Only need one active livestream
                type="video"
            ).execute()
            if not response['items']:
                raise ValueError("No active livestreams found!")
            livestream_id = response['items'][0]['id']['videoId']
            return livestream_id
        else:
            raise ValueError("YouTube channel_id was not provided")

    def get_live_chat_id(self):
        """Retrieve the live chat ID for the current livestream."""
        if self.livestream_id:
            response = self.youtube.videos().list(
                part='liveStreamingDetails',
                id=self.livestream_id
            ).execute()
            live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
            return live_chat_id
        else:
            raise ValueError("No livestream ID provided!")

    async def get_live_chat_messages(self, next_page_token=None):
        """Fetch live chat messages from the current livestream."""
        response = self.youtube.liveChatMessages().list(
            liveChatId=self.live_chat_id,
            part='snippet,authorDetails',
            pageToken=next_page_token
        ).execute()
        messages = response.get('items', [])
        if messages:
            yt_new_messages = [
                f"{message['authorDetails']['displayName']}: {message['snippet']['displayMessage']}"
                for message in messages
            ]
            self.yt_messages.extend(yt_new_messages)
            next_page_token = response.get('nextPageToken')
            dotenv.set_key(
                dotenv_path=dotenv.find_dotenv(),
                key_to_set="LAST_NEXT_PAGE_TOKEN",
                value_to_set=next_page_token
            )
        return next_page_token

if __name__ == "__main__":
    yt_messages = []
    yt_live_controller = YTLive(yt_messages)
    yt_live_controller.get_live_chat_messages()
    print(len(yt_messages), yt_messages)


### LEGACY -- Incase you don't have OAuth2 or don't want to set it up (not recommended)
# from googleapiclient.discovery import build
# from general_utils import get_env_var, retry_with_backoff
# import dotenv



# import os.path
# import pickle
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.discovery import build

# class YTLive():
#     def __init__(self, yt_messages):
#         API_KEY, CHANNEL_ID = self.youtube_credentials()
#         self.youtube = build('youtube', 'v3', developerKey=API_KEY)
#         self.livestream_id = self.get_livestream_id(CHANNEL_ID)
#         self.live_chat_id = self.get_live_chat_id(livestream_id=self.livestream_id)
#         self.yt_messages = yt_messages

#     #retrieves the API key and ChannelID from .env
#     @staticmethod
#     def youtube_credentials():
#         API_KEY, CHANNEL_ID = get_env_var("YT_API_KEY"), get_env_var("YT_CHANNEL_ID")
#         return API_KEY, CHANNEL_ID
    
#     #retrieves livestream ID -- used to fetch live chat id
#     @retry_with_backoff(max_retries=7, initial_delay=5, backoff_factor=2, exceptions=(ValueError,))
#     def get_livestream_id(self, channel_id):
#         """
#         channel_id: Found from YouTube.com -> Settings -> Advanced Settings
#         """
#         if channel_id:
#             #get active livestream data
#             response = self.youtube.search().list(
#                 channelId=channel_id,
#         part="snippet",
#         # eventType="live",
#         maxResults=2,
#         type="video",
#         order="date",
#     ).execute()
#             print(response)
#             if not response['items']:
#                 raise ValueError("No active livestreams found!")

#             livestream_id = response['items'][0]['id']['videoId']
#             return livestream_id
#         else:
#             raise ValueError("YouTube channel_id was not provided")
    
#     #live chat id is used to retrieve live chat messages
#     def get_live_chat_id(self, livestream_id=None):
#         if livestream_id:
#             #get active livestream's live chat id
#             response = self.youtube.videos().list(
#                 part='liveStreamingDetails',
#                 id=livestream_id
#             ).execute()

#             live_chat_id = response['items'][0]['liveStreamingDetails']['activeLiveChatId']
#             return live_chat_id
#         else:
#             raise ValueError("No livestream ID provided!")

#     #fetch youtube livestream chat messages
#     def get_live_chat_messages(self, next_page_token=None):
#         response = self.youtube.liveChatMessages().list(
#             liveChatId=self.live_chat_id,
#             part='snippet,authorDetails',
#             pageToken=next_page_token
#         ).execute()
#         messages = response.get('items', [])
#         if messages:
#             yt_new_messages = [(message['authorDetails']['displayName'],message['snippet']['displayMessage']) for message in messages]
#             self.yt_messages.extend(yt_new_messages)
#             next_page_token = response.get('nextPageToken')
#             #save current next_page_token to ENV variable
#             dotenv.set_key(dotenv_path=dotenv.find_dotenv(),key_to_set="LAST_NEXT_PAGE_TOKEN",value_to_set=next_page_token)
#         return next_page_token
    
# if __name__ == "__main__":
#     yt_messages = []
#     yt_live_controller = YTLive(yt_messages)
#     yt_live_controller.get_live_chat_messages()
#     print(len(yt_messages))
