# youtube.py
import asyncio
import os.path
import pickle
import logging
from typing import List, Optional, Tuple, Dict, Any

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError

# from general_utils import get_env_var
from livechat_utils import retry_with_backoff
# Set up logging
logger = logging.getLogger(__name__)

class YouTubeApiError(Exception):
    """Custom exception for YouTube API-related errors."""
    pass

class YTLive:
    """
    An async, on-demand client to fetch live chat messages from a YouTube livestream.
    This client is designed to be initialized, set up once, and then polled.
    """
    def __init__(self, channel_id: str, client_secret_file: str):
        """
        channel_id (str): The YouTube channel ID.
        client_secret_file (str): The path to the client secret JSON file. Usually downloaded from the YouTube developer console control panel.
        """
        # self.channel_id = get_env_var("YT_CHANNEL_ID", var_type=str)
        # if not self.channel_id:
        #     raise ValueError("YT_CHANNEL_ID environment variable is not set.")
        if not channel_id:
            raise ValueError("YouTube channel_id cannot be empty.")
        if not client_secret_file:
            raise ValueError("YouTube client_secret_file path cannot be empty.")
            
        self.channel_id = channel_id
        self.client_secret_file = client_secret_file
        
        self._creds = self._load_credentials()
        self.youtube: Resource = build('youtube', 'v3', credentials=self._creds)
        
        self.livestream_id: str = None
        self.live_chat_id: str = None

    def _load_credentials(self):
        """Loads or refreshes user credentials for OAuth2."""
        SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
        creds = None
        
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logging.info("Refreshing expired YouTube credentials.")
                creds.refresh(Request())
            else:
                logging.info("Fetching new YouTube credentials via OAuth flow.")
                if not os.path.exists(self.client_secret_file):
                    raise FileNotFoundError(f"YouTube client secret JSON not found at path: {self.client_secret_file}")
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        logging.info("YouTube credentials loaded successfully.")
        return creds

    async def setup(self):
        """
        Asynchronously fetches the livestream and chat IDs.
        This should be called once after initialization.
        """
        logging.info("Setting up YTLive client...")
        self.livestream_id = await self._get_livestream_id()
        self.live_chat_id = await self._get_live_chat_id()
        if self.live_chat_id:
            logging.info(f"Successfully setup YouTube client. Live Chat ID: {self.live_chat_id}")

    @retry_with_backoff(max_retries=5, initial_delay=5, backoff_factor=2, exceptions=(YouTubeApiError,))
    async def _get_livestream_id(self) -> str:
        """Asynchronously retrieve the ID of an active livestream."""
        def _execute_request():
            return self.youtube.search().list(
                channelId=self.channel_id,
                part="snippet",
                eventType="live",
                maxResults=1,
                type="video"
            ).execute()

        try:
            response = await asyncio.to_thread(_execute_request)
            if not response.get('items'):
                raise YouTubeApiError("No active livestreams found for the channel.")
            return response['items'][0]['id']['videoId']
        except HttpError as e:
            logging.error(f"HTTP error while fetching livestream ID: {e}")
            raise YouTubeApiError(f"HTTP error fetching livestream ID: {e.status_code}") from e

    async def _get_live_chat_id(self) -> Optional[str]:
        """Asynchronously retrieve the live chat ID for the current livestream."""
        if not self.livestream_id:
            raise YouTubeApiError("Cannot get chat ID without a livestream ID.")

        def _execute_request():
            return self.youtube.videos().list(
                part='liveStreamingDetails',
                id=self.livestream_id
            ).execute()
        
        try:
            response = await asyncio.to_thread(_execute_request)
            if not response.get('items') or 'liveStreamingDetails' not in response['items'][0]:
                logging.warning("Livestream found, but it has no live chat details. It may have ended.")
                return None
            return response['items'][0]['liveStreamingDetails']['activeLiveChatId']
        except HttpError as e:
            logging.error(f"HTTP error while fetching live chat ID: {e}")
            raise YouTubeApiError(f"HTTP error fetching live chat ID: {e.status_code}") from e

    async def get_live_chat_messages(self, next_page_token: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Fetches a page of live chat messages asynchronously.

        Args:
            next_page_token: The token for the next page of results.

        Returns:
            A tuple containing a list of raw message items and the next page token.
        """
        if not self.live_chat_id:
            logging.warning("No live chat ID available. Cannot fetch messages.")
            return [], None

        def _execute_request():
            return self.youtube.liveChatMessages().list(
                liveChatId=self.live_chat_id,
                part='snippet,authorDetails',
                pageToken=next_page_token
            ).execute()

        try:
            response = await asyncio.to_thread(_execute_request)
            messages = response.get('items', [])
            new_next_page_token = response.get('nextPageToken')
            return messages, new_next_page_token
        except HttpError as e:
            # A 403 error often means the chat has ended. This is not a critical failure.
            if e.status_code == 403:
                logging.warning(f"Could not fetch YouTube chat messages (HTTP 403). The livestream or chat may have ended.")
                return [], None
            logging.error(f"HTTP error while fetching live chat messages: {e}")
            return [], None # Return empty list to be resilient

### The section below is for standalone testing.

async def _test_youtube_client():
    """Demonstrates the new on-demand, async usage of the YTLive client."""
    print("--- Running YTLive Standalone Test ---")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from dotenv import load_dotenv
    load_dotenv()
    from dotenv import load_dotenv
    from general_utils import get_env_var
    load_dotenv()

    try:
        # Configuration is loaded and passed to the client's constructor.
        channel_id = get_env_var("YT_CHANNEL_ID", var_type=str)
        secret_file = get_env_var("YT_OAUTH2_JSON", var_type=str)
        
        client = YTLive(channel_id=channel_id, client_secret_file=secret_file)
        await client.setup()

        if client.live_chat_id:
            print(f"Successfully connected to live chat: {client.live_chat_id}")
            
            test_page_token = None 
            
            print("\nFetching first page of messages...")
            messages, next_page_token = await client.get_live_chat_messages(test_page_token)
            
            if messages:
                for msg in messages:
                    print(f"  -> {msg['authorDetails']['displayName']}: {msg['snippet']['displayMessage']}")
                print(f"\nSuccessfully fetched {len(messages)} messages.")
                print(f"Next page token is: {next_page_token}")
            else:
                print("No new messages found on this poll.")
        else:
            print("Could not set up YouTube client. Check logs for errors.")

    except (ValueError, FileNotFoundError, YouTubeApiError) as e:
        print(f"🚨 An error occurred during the test: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(_test_youtube_client())
    except KeyboardInterrupt:
        print("\nExiting test.")

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
