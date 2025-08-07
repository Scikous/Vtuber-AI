import time
from functools import wraps

#used as a decorator for functions that can fail initially, namely YouTube/Twitch/Kick livechat

# def retry_with_backoff(max_retries=3, initial_delay=5, backoff_factor=2, exceptions=(Exception,)):
#     """
#     A decorator that retries the decorated function with exponential backoff.

#     :param max_retries: Maximum number of retries before giving up
#     :param initial_delay: Initial delay between retries in seconds
#     :param backoff_factor: Multiplier for delay after each retry
#     :param exceptions: Tuple of exceptions to catch and retry on
#     """
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             delay = initial_delay

#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except exceptions as e:
#                     if attempt == max_retries - 1:
#                         raise e
#                     print(f"Attempt {attempt + 1} failed with the following ERROR: {e}. Retrying in {delay} seconds...")
#                     time.sleep(delay)
#                     delay *= backoff_factor
#         return wrapper
#     return decorator

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

