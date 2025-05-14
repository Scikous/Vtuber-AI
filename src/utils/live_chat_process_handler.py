import asyncio
import logging
import sys
import os

# Attempt to import LiveChatController, with fallback for PYTHONPATH issues
try:
    from Livechat_Wizard.livechat import LiveChatController
except ImportError:
    # This block tries to add the project root to sys.path if LiveChatController is not found.
    # Assumes this script is in src/utils/ and project root is two levels up.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from Livechat_Wizard.livechat import LiveChatController
    except ImportError as e:
        # If still not found, log and re-raise or set LiveChatController to None
        # For now, we'll let it fail if not found after trying to adjust path,
        # as it's a critical dependency for this module.
        logging.basicConfig(level=logging.ERROR) # Ensure logging is configured for this message
        logging.error(f"Failed to import LiveChatController even after sys.path adjustment: {e}")
        LiveChatController = None # Allow the rest of the file to be parsed, but it will fail at runtime

def live_chat_process_target(mp_queue):
    """
    Target function for the live chat multiprocessing.Process.
    Fetches messages from LiveChatController and puts them into mp_queue.
    """
    # Basic logging setup for this process.
    # In a more complex app, you might pass a logging configuration dict.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s LiveChatProcess: %(message)s')
    logger = logging.getLogger("LiveChatProcess")

    if LiveChatController is None:
        logger.error("LiveChatController could not be imported. Live chat process cannot start.")
        mp_queue.put(None) # Signal failure
        return

    logger.info("Live chat process started.")

    try:
        live_chat_controller = LiveChatController.create()
    except Exception as e:
        logger.error(f"Failed to create LiveChatController: {e}", exc_info=True)
        mp_queue.put(None) # Signal failure
        return

    if not live_chat_controller:
        mp_queue.put(None)  # Signal that no live chat is available (e.g., not configured)
        logger.info("Live chat functionality is not available (LiveChatController.create() returned None).")
        return

    logger.info("LiveChatController created successfully.")

    async def fetch_and_send():
        logger.info("Starting fetch_and_send async loop inside live_chat_process_target.")
        fetch_interval = 15.0 # Default fetch interval in seconds
        if hasattr(live_chat_controller, 'get_fetch_interval'):
            try:
                fetch_interval = float(live_chat_controller.get_fetch_interval())
            except (ValueError, TypeError):
                logger.warning(f"Could not get valid fetch interval from controller, using default {fetch_interval}s.")

        while True:
            try:
                # fetch_chat_message() is expected to be an async method
                live_chat_msg = await asyncio.wait_for(live_chat_controller.fetch_chat_message(), timeout=10.0)
                if live_chat_msg:
                    # Assuming live_chat_msg is a tuple (author, message)
                    formatted_msg = f"{live_chat_msg[0]}: {live_chat_msg[1]}"
                    mp_queue.put(formatted_msg)
                    logger.debug(f"Sent to mp_queue: {formatted_msg}")
            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for live chat message.")
                pass # Continue to the next iteration
            except Exception as e:
                logger.error(f"Error in fetch_and_send loop: {e}", exc_info=True)
                break # Exit loop on error
            
            await asyncio.sleep(fetch_interval)

    try:
        asyncio.run(fetch_and_send())
    except Exception as e:
        logger.error(f"Critical error running asyncio loop in live_chat_process_target: {e}", exc_info=True)
    finally:
        logger.info("Live chat process target function finished.")
        # Optionally, put a final None or specific sentinel to indicate clean shutdown if mp_queue is still used by consumer
        # mp_queue.put(None) # This might be handled by the consumer checking process liveness