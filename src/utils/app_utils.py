"""
Application Utilities for Vtuber-AI
Provides miscellaneous utility functions and decorators for the application.
"""
import sys
import os
import time
import asyncio
from functools import wraps
from contextlib import contextmanager

@contextmanager
def change_dir(new_dir):
    """Context manager to temporarily change the current working directory."""
    old_dir = os.getcwd()
    os.makedirs(new_dir, exist_ok=True) # Ensure the new directory exists
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)

def retry_with_backoff(max_retries=3, initial_delay=5, backoff_factor=2, exceptions=(Exception,)):
    """
    A decorator that retries the decorated function with exponential backoff.

    :param max_retries: Maximum number of retries before giving up
    :param initial_delay: Initial delay between retries in seconds
    :param backoff_factor: Multiplier for delay after each retry
    :param exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # self.logger.error(f"Max retries reached for {func.__name__}. Last error: {e}") # Assuming logger is available
                        print(f"Max retries reached for {func.__name__}. Last error: {e}")
                        raise e
                    # self.logger.warning(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
                    print(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # self.logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
                        print(f"Max retries reached for {func.__name__}. Last error: {e}")
                        raise e
                    # self.logger.warning(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
                    print(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def setup_project_root():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)