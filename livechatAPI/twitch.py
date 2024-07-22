import requests, json, webbrowser
from flask import Flask, request, redirect
from requests_oauthlib import OAuth2Session
import os, time
import threading
from livechat_utils import append_message, write_messages_csv

CHANNEL = 'scikous'
BOT_NICK = 'Botty'
TOKEN_URL = 'https://id.twitch.tv/oauth2/token'  # Refresh token endpoint
REDIRECT_URI = 'https://localhost:8080'
AUTHORIZATION_BASE_URL = 'https://id.twitch.tv/oauth2/authorize'

twitch_chat_msgs = []
DEFAULT_SAVE_FILE = "livechatAPI/data/twitch_chat.csv"


class TwitchTools():
    @staticmethod
    def twitch_auth_loader(cred_file):
        with open(cred_file, 'r') as credentials:
            creds = json.load(credentials)
            CLIENT_ID = creds["client-id"]
            CLIENT_SECRET = creds["client-secret"]
        return CLIENT_ID, CLIENT_SECRET

app = Flask(__name__)
class TwitchAuth():
    def __init__(self, CLIENT_ID, CLIENT_SECRET) -> None:
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET
        self.oauth = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["chat:read", "chat:edit"])

        @app.route('/')
        def index():
            # Only redirect to the authorization URL once
            if 'code' in request.args or 'state' in request.args:
                token = self.oauth.fetch_token(TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET,include_client_id=True, authorization_response=request.url)
                #print(f"Token received: {token}")
                # Save the token to a file or environment variable
                with open('token.json', 'w') as token_file:
                    json.dump(token, token_file)
                return 'Authorization successful! You can close this window.'
                # return redirect('/callback')
            else:
                authorization_url, state = self.oauth.authorization_url(AUTHORIZATION_BASE_URL)
                return redirect(authorization_url)
    @staticmethod
    def validate_token(expiration):
        current_time = time.time()
        if expiration - current_time < 500:
            return True
        return False
    def token_from_file(self, file_path):
        with open('token.json', 'r') as token_file:
            token_data = json.load(token_file)
            expired = self.validate_token(token_data['expires_at'])
            token = token_data["access_token"] if not expired else None
        return token
    
    @staticmethod
    def run_flask(stop_flask):
        while not stop_flask.is_set():
            # Start the Flask server
            app.run(port=8080, ssl_context='adhoc')
    @staticmethod
    def stop_flask(stop_event: threading.Event):
        print("Shutting down flask server")
        stop_event.set()

    def auth_access_token(self):
            # Load the token from the file
        try:
            token = self.token_from_file('token.json')
            if not token:
                stop_event = threading.Event()
                flask_thread = threading.Thread(target=self.run_flask, args=(stop_event,), daemon=True)
                flask_thread.start()
                # Open the authorization URL in the browser
                webbrowser.open(f'https://localhost:8080')
                # Wait for the Flask server to handle the redirect and save the token
                time.sleep(15)  # Adjust the sleep time if necessary

                # Print the access token
                print('\n' * 25 + '#' * 30)
                self.stop_flask(stop_event)
                flask_thread.join()
                print("flask server exited")
                token = self.token_from_file('token.json')
            return token
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
    def __init__(self, TOKEN, CLIENT_ID, BOT_NICK, CHANNEL, SAVE_MESSAGES=True):
        super().__init__(token=TOKEN, client_id=CLIENT_ID, nick=BOT_NICK, prefix='!', initial_channels=[CHANNEL])
        self.write_message_func = write_messages_csv if SAVE_MESSAGES else None  # Assign conditionally
    async def event_ready(self):
        print(f'Ready | {self.nick}')

    async def event_message(self, message):
        print(f'{message.author.name}: {message.content}')
        if '!' in message.content:
            await self.handle_commands(message)
        user_msg = (message.author.name, message.content)
        #write message to file -- stability is questionable for bigger stream chats
        if self.write_message_func:
            self.write_message_func(DEFAULT_SAVE_FILE, user_msg)
        append_message(twitch_chat_msgs, user_msg)
        
    @commands.command(name='hello')
    async def hello(self, ctx):
        await ctx.send(f'Hello {ctx.author.name}!')



if __name__ == "__main__":
    TWTools = TwitchTools()
    CLIENT_ID, CLIENT_SECRET = TWTools.twitch_auth_loader("livechatAPI/credentials/twitch.json")
    TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)

    # # Replace with your Twitch token and channel
    TOKEN = TW_Auth.auth_access_token()             
    bot = Bot(TOKEN,CLIENT_ID, BOT_NICK, CHANNEL)
    bot.run()
    