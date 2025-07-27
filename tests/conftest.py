import sys
import os
from dotenv import load_dotenv

# Add the project's root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()#get .env file variables
