import requests, json, webbrowser
from flask import Flask, request, redirect
from requests_oauthlib import OAuth2Session
import os, time
import threading
from livechat_utils import append_livechat_message, write_messages_csv

TOKEN_URL = 'https://id.twitch.tv/oauth2/token'  # Refresh token endpoint
REDIRECT_URI = 'https://localhost:8080'
AUTHORIZATION_BASE_URL = 'https://id.twitch.tv/oauth2/authorize'

twitch_chat_msgs = []
DEFAULT_SAVE_FILE = "livechatAPI/data/twitch_chat.csv"

class TwitchAuth():
    def __init__(self, credentials_file="livechatAPI/credentials/twitch.json") -> None:
        self._token_file = "livechatAPI/credentials/twitch_token.json"
        self.CHANNEL, self.BOT_NICK, self.CLIENT_ID, self.CLIENT_SECRET, self.TOKEN, = self.twitch_auth_loader(credentials_file)
        self._oauth = OAuth2Session(self.CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["chat:read", "chat:edit"])
        self._stop_event = threading.Event()
        self._flask_thread = None

    def twitch_auth_loader(self, credentials_file="livechatAPI/credentials/twitch.json"):
        with open(credentials_file, 'r') as credentials:
            creds = json.load(credentials)
            CHANNEL = creds["channel"]
            BOT_NICK = creds["bot-nick"]
            CLIENT_ID = creds["client-id"]
            CLIENT_SECRET = creds["client-secret"]
            TOKEN = self.token_from_file()
        return CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN
        
    def token_from_file(self):
        def _validate_token(expiration):
            current_time = time.time()
            if expiration - current_time < 500:
                return True
            return False
        try:
            with open(self._token_file, 'r') as token_file:
                token_data = json.load(token_file)
                expired = _validate_token(token_data['expires_at'])
                token = token_data["access_token"] if not expired else None
            return token
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading token (probably empty file): {e}")
            return None
        
    def stop_flask(self):
        print("Shutting down flask server")
        self._stop_event.set()
        self._flask_thread.join()
        print("flask server exited")
    
    def run_flask(self):
        app = Flask(__name__)
        @app.route('/')
        def index():
            # Only redirect to the authorization URL once
            if 'code' in request.args or 'state' in request.args:
                token = self._oauth.fetch_token(TOKEN_URL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET,include_client_id=True, authorization_response=request.url)
                #print(f"Token received: {token}")
                # Save the token to a file or environment variable
                with open(self._token_file, 'w') as token_file:
                    json.dump(token, token_file)
                self.stop_flask()
                return 'Authorization successful! You can close this window.'
            else:
                authorization_url, state = self._oauth.authorization_url(AUTHORIZATION_BASE_URL)
                return redirect(authorization_url)
        app.run(port=8080, ssl_context='adhoc')


    def auth_access_token(self):
        try: 
            self._flask_thread = threading.Thread(target=self.run_flask, daemon=True)
            self._flask_thread.start()

            # Open the authorization URL in the browser
            webbrowser.open(f'https://localhost:8080')
            
            # Wait for the Flask server to handle the authorization redirect, save the token and shutdown the flask server
            self._stop_event.wait()
            
            self.TOKEN = self.token_from_file()
            return self.TOKEN
        except Exception as e:
            print(f'HTTPError: {e}')

    def refresh_access_token(self,refresh_token):
    # Prepare the request data
        data = {
            'grant_type': 'refresh_token',
            'client_id': self.CLIENT_ID,
            'client_secret': self.CLIENT_SECRET,
            'refresh_token': refresh_token
        }

        # Send a POST request to the refresh token endpoint
        response = requests.post(TOKEN_URL, data=data)

        # Check for successful response
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()
            return response_data.get('access_token')  # Extract the new access token
        else:
            # Handle error
            print(f'Error refreshing access token: {response.status_code}')
            return None
# print(auth_token)
# Example usage (replace with your credentials)


#########wip
# refresh_token = 'your_refresh_token'  # Obtained during initial authorization

# new_access_token = TwitchAuth.refresh_access_token(CLIENT_ID, CLIENT_SECRET, refresh_token)

# if new_access_token:
#     print('Successfully refreshed access token')
# # Use the new_access_token for your Twitch API calls
# else:
#     print('Failed to refresh access token')
#######



from twitchio.ext import commands
class Bot(commands.Bot):
    def __init__(self, CHANNEL, BOT_NICK, CLIENT_ID, TOKEN, SAVE_MESSAGES=True):
        super().__init__(token=TOKEN, client_id=CLIENT_ID, nick=BOT_NICK, prefix='!', initial_channels=[CHANNEL])
        self.write_message_func = write_messages_csv if SAVE_MESSAGES else None  # Assign conditionally
    async def event_ready(self):
        print(f'Ready | {self.nick}')

    async def event_message(self, message):
        print(f'{message.author.name}: {message.content}')
        if '!' in message.content:
            await self.handle_commands(message)
        user_msg = (message.author.name, message.content)
        print("Twitch msg:", user_msg)
        #write message to file -- stability is questionable for bigger stream chats
        if self.write_message_func:
            self.write_message_func(DEFAULT_SAVE_FILE, user_msg)
        append_livechat_message(twitch_chat_msgs, user_msg)
        
    @commands.command(name='hello')
    async def hello(self, ctx):
        await ctx.send(f'Hello {ctx.author.name}!')



if __name__ == "__main__":

    # # Replace with your Twitch token and channel
    TW_Auth = TwitchAuth()
    CHANNEL, BOT_NICK, CLIENT_ID, TOKEN = TW_Auth.CHANNEL, TW_Auth.BOT_NICK, TW_Auth.CLIENT_ID, TW_Auth.TOKEN
    if not TOKEN:
        TOKEN = TW_Auth.auth_access_token()
    bot = Bot(CHANNEL, BOT_NICK, CLIENT_ID, TOKEN)
    bot.run()
    