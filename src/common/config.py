"""
Configuration Loader for Vtuber-AI
Provides methods to load and access shared configuration settings.
"""
import os
import json

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            from utils.logger import conditional_print
            conditional_print(f"Using default config path: {config_path}")
        except ImportError:
            # Fallback to regular print if conditional_print is not available
            print(f"Using default config path: {config_path}")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)