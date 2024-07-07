import requests, json, webbrowser
from flask import Flask, request, redirect
from requests_oauthlib import OAuth2Session
import os, time
import threading


TOKEN_URL = 'https://id.twitch.tv/oauth2/token'  # Refresh token endpoint
REDIRECT_URI = 'https://localhost:8080'
AUTHORIZATION_BASE_URL = 'https://id.twitch.tv/oauth2/authorize'
app = Flask(__name__)

def twitch_auth_loader(cred_file):
    with open(cred_file, 'r') as credentials:
        creds = json.load(credentials)
        CLIENT_ID = creds["client-id"]
        CLIENT_SECRET = creds["client-secret"]
    return CLIENT_ID, CLIENT_SECRET

class TwitchAuth():
    def __init__(self, CLIENT_ID, CLIENT_SECRET) -> None:
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET
        self.oauth = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["chat:read", "chat:edit"])

        @app.route('/')
        def index():
            # Only redirect to the authorization URL once
            if 'code' in request.args or 'state' in request.args:
                token_fetch_url = 'https'+ request.url[4:]
                print(token_fetch_url, request.url) 
                token = self.oauth.fetch_token(TOKEN_URL, client_id=CLIENT_ID, client_secret=CLIENT_SECRET,include_client_id=True, authorization_response=request.url)
                print(f"Token received: {token}")
                # Save the token to a file or environment variable
                with open('token.json', 'w') as token_file:
                    json.dump(token, token_file)
                return 'Authorization successful! You can close this window.'
                # return redirect('/callback')
            else:
                authorization_url, state = self.oauth.authorization_url(AUTHORIZATION_BASE_URL)
                return redirect(authorization_url)

    def auth_access_token(self):
        try:
            # Start the Flask server in a separate process
            # Open the authorization URL in the browser
            webbrowser.open(f'https://localhost:8080')
            # Wait for the Flask server to handle the redirect and save the token
            time.sleep(30)  # Adjust the sleep time if necessary

            # Load the token from the file
            with open('token.json', 'r') as token_file:
                token = json.load(token_file)

            # Print the access token
            print('\n' * 25 + '#' * 30)
            return token["access_token"]
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

CLIENT_ID, CLIENT_SECRET = twitch_auth_loader("livechatAPI/credentials/twitch.json")

TW_Auth = TwitchAuth(CLIENT_ID, CLIENT_SECRET)


auth_thread = threading.Thread(target=app.run(port=8080, ssl_context='adhoc'), daemon=True)
auth_thread.start()
TW_Auth.auth_access_token()
# Example usage (replace with your credentials)





# refresh_token = 'your_refresh_token'  # Obtained during initial authorization

# new_access_token = TwitchAuth.refresh_access_token(CLIENT_ID, CLIENT_SECRET, refresh_token)

# if new_access_token:
#     print('Successfully refreshed access token')
# # Use the new_access_token for your Twitch API calls
# else:
#     print('Failed to refresh access token')
