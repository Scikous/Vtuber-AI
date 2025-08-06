"""
Logger Utility for Vtuber-AI
Provides a centralized, process-safe logger for the application.
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# --- Configuration ---
# Use an absolute path to the project root to ensure files are always found
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'src', 'common', 'config.json')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

def setup_logging():
    """
    Configures the root logger for a single process.
    This function is idempotent and safe to call multiple times.
    It reads the configuration to decide whether to enable logging.
    """
    root_logger = logging.getLogger()
    
    # If handlers are already configured, another part of the process did it. Don't add more.
    if root_logger.hasHandlers():
        return

    # --- Determine if logging should be enabled ---
    logging_enabled = False
    try:
        import json
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logging_enabled = config.get('logging_enabled', False)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Fallback for situations where config is not available
        print(f"[PRE-LOGGING WARNING] Could not load config to check logging status: {e}", file=sys.stderr)
        logging_enabled = os.environ.get('LOGGING_ENABLED', 'false').lower() == 'true'

    # --- Set the master level for the root logger ---
    # If disabled, we set the level so high that nothing gets through.
    # If enabled, we set it to DEBUG to capture everything, and let handlers filter.
    log_level = logging.DEBUG if logging_enabled else logging.CRITICAL + 1
    root_logger.setLevel(log_level)

    # --- Create Formatter ---
    # Include process info, which is crucial for debugging multiprocessing apps
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] (%(processName)s:%(process)d) - %(message)s'
    )

    # --- Configure Handlers (only if logging is enabled) ---
    if logging_enabled:
        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO) # Console shows INFO and above
        root_logger.addHandler(stream_handler)

        # File Handler (with rotation)
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file_path = os.path.join(LOG_DIR, 'app.log')
        
        # Rotates logs after 5MB, keeping 3 backup files.
        file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG) # Log file captures everything (DEBUG and above)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging configured for process {os.getpid()}. Level: {logging.getLevelName(log_level)}. Log file: {log_file_path}")
    else:
        # Add a NullHandler to prevent "No handlers could be found" warnings
        root_logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance. It will inherit the root configuration.
    """
    return logging.getLogger(name)