import requests, webbrowser
from flask import Flask, request, redirect
from requests_oauthlib import OAuth2Session
import multiprocessing
from livechat_utils import append_livechat_message, write_messages_csv, get_env_var
import dotenv

twitch_chat_msgs = []

class TwitchAuth():
    def __init__(self) -> None:
        self.CHANNEL, self.BOT_NICK, self.CLIENT_ID, self.CLIENT_SECRET, self.ACCESS_TOKEN, self.USE_THIRD_PARTY_TOKEN = self.twitch_auth_loader()
        self.TOKEN_URL = 'https://id.twitch.tv/oauth2/token'  # Refresh token endpoint

    #load in necessary twitch credentials from .env
    def twitch_auth_loader(self):
        CHANNEL, BOT_NICK = get_env_var("TW_CHANNEL"), get_env_var("TW_BOT_NICK")
        CLIENT_ID, CLIENT_SECRET = get_env_var("TW_CLIENT_ID"), get_env_var("TW_CLIENT_SECRET")
        ACCESS_TOKEN = get_env_var("TW_ACCESS_TOKEN") #technically the refresh token if LOCAL GENERATION = True, but don't worry about it
        USE_THIRD_PARTY_TOKEN = get_env_var("TW_USE_THIRD_PARTY_TOKEN") == 'True' #if not locally generating token, handle access token differently 
        return CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, USE_THIRD_PARTY_TOKEN

    #turn off the flask server when done with it -- may or may not be working
    def stop_flask(self, process_stop_event):
        print("Flask server shutdown signal sent")
        process_stop_event.set()

    #create a temporary HTTPS flask server which can be used to generate tokens
    def run_flask(self, token, process_stop_event):
        REDIRECT_URI = 'https://localhost:8080'
        AUTHORIZATION_BASE_URL = 'https://id.twitch.tv/oauth2/authorize'
        app = Flask(__name__)
        oauth = OAuth2Session(self.CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["chat:read", "chat:edit"])
        @app.route('/')
        def index():
            # Only redirect to the authorization URL once
            if 'code' in request.args or 'state' in request.args:
                token_info = oauth.fetch_token(self.TOKEN_URL, client_id=self.CLIENT_ID, client_secret=self.CLIENT_SECRET,include_client_id=True, authorization_response=request.url)
                #print(f"Token received: {token}")
                
                #sets process token to access token directly -- avoids having to call refresh method
                token.value = bytes(token_info['access_token'],'utf-8')

                #Save the token to a file or environment variable
                dotenv.set_key(dotenv_path=".env",key_to_set="TW_ACCESS_TOKEN",value_to_set=token_info["refresh_token"])
                # with open('w') as token_file:
                    # json.dump(token, token_file)
                self.stop_flask(process_stop_event)
                return 'Authorization successful! You can close this window.'
            else:
                authorization_url, state = oauth.authorization_url(AUTHORIZATION_BASE_URL)
                return redirect(authorization_url)
        app.run(port=8080, ssl_context='adhoc')

    #a generator for tokens
    def access_token_generator(self):
        """
        OAuth2 requires authorization, so a web browser is needed, and a web page will be opened.
        The web page is "insecure" and requires manual authorization.
        """
        try: 
            process_stop_event = multiprocessing.Event()
            #create an empty character array of 30 bytes -- processes can't by default share memory
            token = multiprocessing.Array('c', 30)
            
            #create a temporary process to handle flask server
            flask_process = multiprocessing.Process(target=self.run_flask, args=(token, process_stop_event),daemon=True)
            flask_process.start()

            # Open the authorization URL in the browser
            webbrowser.open(f'https://localhost:8080')
            
            # Wait for the Flask server to handle the authorization redirect, save the token and shutdown the flask server
            process_stop_event.wait()
            flask_process.terminate()
            flask_process.join()
            print("flask server exited")
            return token.value.decode('utf-8') #decode bytes -> str
        except Exception as e:
            print(f'HTTPError: {e}')

    #once the first token has been generated, the refresh token can be used to generate new ones
    def refresh_access_token(self):
    # Prepare the request data
        data = {
            'grant_type': 'refresh_token',
            'client_id': self.CLIENT_ID,
            'client_secret': self.CLIENT_SECRET,
            'refresh_token': self.ACCESS_TOKEN
        }

        # Send a POST request to the refresh token endpoint
        response = requests.post(self.TOKEN_URL, data=data)

        # Check for successful response
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()
            return response_data.get('access_token')  # Extract the new access token
        else:
            # Handle error
            print(f'Error refreshing access token: {response.status_code}')
            return None

from twitchio.ext import commands
class Bot(commands.Bot):
    """
    A twitchio bot which is used to primarily retrieve and "forward" messages to the LLM.
    """
    def __init__(self, CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN):
        super().__init__(token=TOKEN, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, nick=BOT_NICK, prefix='!', initial_channels=[CHANNEL])

    async def event_ready(self):
        print(f'Ready | {self.nick}')

    #upon receiving a message, username and message are appended to a list for the LLM to pick from
    async def event_message(self, message):
        # print(f'{message.author.name}: {message.content}')
        if '!' in message.content:
            await self.handle_commands(message)
        user_msg = (message.author.name, message.content)
        # print("Twitch msg:", user_msg)
        append_livechat_message(twitch_chat_msgs, user_msg)

    @commands.command(name='hello')
    async def hello(self, ctx):
        await ctx.send(f'Hello {ctx.author.name}!')
    
#For testing purposes
if __name__ == "__main__":
    # Replace with your Twitch token and channel
    dotenv.load_dotenv()
    import twitchio
    TW_Auth = TwitchAuth()
    CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, USE_THIRD_PARTY_TOKEN = TW_Auth.CHANNEL, TW_Auth.BOT_NICK, TW_Auth.CLIENT_ID, TW_Auth.CLIENT_SECRET, TW_Auth.ACCESS_TOKEN, TW_Auth.USE_THIRD_PARTY_TOKEN
    if USE_THIRD_PARTY_TOKEN:
        TOKEN = ACCESS_TOKEN
    elif not ACCESS_TOKEN:
        TOKEN = TW_Auth.access_token_generator()
    else:
        TOKEN = TW_Auth.refresh_access_token()

    try:
        bot = Bot(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN)
        bot.run()
    except twitchio.AuthenticationError as e:
        print("TOKEN EXPIRED LAMO", e)
        TOKEN = TW_Auth.refresh_access_token()
        bot = Bot(CHANNEL, BOT_NICK, CLIENT_ID, CLIENT_SECRET, TOKEN)
        bot.run()
    except KeyError as e:
        print("TOKEN EXPIRED with KeyERROR, LMAO", e)    
    except ValueError as e:
        print("TOKEN EXPIRED with ValueERROR, LMAO", e)    
    except Exception as e:
        print("TOKEN EXPIRED with Exception, LMAO", e)