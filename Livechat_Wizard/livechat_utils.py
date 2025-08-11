import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

import asyncio
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
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
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
                        logger.error(f"Max retries reached for {func.__name__}. Last error: {e}")
                        raise e
                    logger.warning(f"Attempt {attempt + 1} for {func.__name__} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= backoff_factor
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

