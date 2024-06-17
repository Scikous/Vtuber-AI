import json

def api_key_loader(cred_file):
    with open(cred_file, 'r') as credentials:
        creds = json.load(credentials)
        api_key = creds["api_key"]
    return api_key

youtube_creds = api_key_loader('livechatAPI/credentials/youtube.json')
