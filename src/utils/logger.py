"""
Logger Utility for Vtuber-AI
Provides a centralized logger for the application.
"""
import logging
import sys
import os

# Global flag to control logging
_logging_enabled = None

def _check_logging_enabled():
    """Check if logging is enabled from config or environment variable."""
    global _logging_enabled
    if _logging_enabled is None:
        # Try to load from config first
        try:
            from common import config as app_config
            config = app_config.load_config()
            _logging_enabled = config.get('logging_enabled', False)
        except:
            # Fallback to environment variable
            _logging_enabled = os.environ.get('LOGGING_ENABLED', 'false').lower() == 'true'
    return _logging_enabled

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Override logger methods to check logging enabled flag
    original_info = logger.info
    original_debug = logger.debug
    original_warning = logger.warning
    original_error = logger.error
    original_critical = logger.critical
    
    def conditional_info(msg, *args, **kwargs):
        if _check_logging_enabled():
            original_info(msg, *args, **kwargs)
    
    def conditional_debug(msg, *args, **kwargs):
        if _check_logging_enabled():
            original_debug(msg, *args, **kwargs)
    
    def conditional_warning(msg, *args, **kwargs):
        if _check_logging_enabled():
            original_warning(msg, *args, **kwargs)
    
    def conditional_error(msg, *args, **kwargs):
        if _check_logging_enabled():
            original_error(msg, *args, **kwargs)
    
    def conditional_critical(msg, *args, **kwargs):
        if _check_logging_enabled():
            original_critical(msg, *args, **kwargs)
    
    logger.info = conditional_info
    logger.debug = conditional_debug
    logger.warning = conditional_warning
    logger.error = conditional_error
    logger.critical = conditional_critical
    
    return logger

def conditional_print(*args, **kwargs):
    """Print function that only prints when logging is enabled."""
    if _check_logging_enabled():
        print(*args, **kwargs)