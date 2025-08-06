# src/workers/livechat_worker.py

import asyncio
import multiprocessing as mp
# It's crucial to set up the project root to allow imports from other directories
from src.utils.env_utils import setup_project_root
setup_project_root()

from src.utils import logger as app_logger
from src.common import config as app_config
from Livechat_Wizard.livechat import LiveChatController
from Livechat_Wizard.data_models import UnifiedMessage # Assuming data_models.py is accessible from the project root


# Get a logger for this worker
app_logger.setup_logging()
logger = app_logger.get_logger("LiveChatWorker")

async def runner(shutdown_event: mp.Event, toggle_event: mp.Event, output_queue: mp.Queue):
    """
    The asynchronous core of the live chat worker.
    Initializes the controller and runs the main fetch loop.
    """
    
    # Load configuration to get the fetch interval
    config = app_config.load_config()
    fetch_interval_s = config.get("livechat_settings", {}).get("fetch_interval_s", 15)
    logger.info(f"Live chat fetch interval set to {fetch_interval_s} seconds.")

    # Create the controller using the factory method which checks .env variables
    controller = LiveChatController.create()
    if not controller:
        logger.error("LiveChatController could not be created. Check .env configuration (e.g., YT_FETCH). Worker will exit.")
        return

    await controller.setup_clients()
    await controller.start_services()

    try:
        logger.info("LiveChat Worker is running and waiting for toggle.")
        while not shutdown_event.is_set():
            # The worker only fetches if the toggle event is set from the UI
            if not toggle_event.is_set():
                # Sleep for a short duration to avoid a busy-wait loop while disabled
                await asyncio.sleep(1)
                continue

            try:
                logger.debug("Fetching new chat message...")
                winner_message, context_messages = await controller.fetch_chat_message()
                

                if winner_message:
                    logger.info(f"Got a winning message from {winner_message.platform} with {len(context_messages)} context messages. Placing in queue.")
                    
                    #extra messages may be helpful, especially for LLM to avoid false positives during appropriateness check
                    payload = {
                        "winner": winner_message,
                        "context": context_messages
                    }
                    output_queue.put(payload)
                
            except Exception as e:
                logger.error(f"An error occurred during the fetch cycle: {e}", exc_info=True)
            
            # Wait for the configured interval before the next fetch cycle
            await asyncio.sleep(fetch_interval_s)

    except asyncio.CancelledError:
        logger.info("Runner task was cancelled.")
    finally:
        logger.info("Shutting down LiveChatController services...")
        await controller.stop_services()
        logger.info("LiveChat Worker has shut down.")


def livechat_worker(shutdown_event: mp.Event, toggle_event: mp.Event, output_queue: mp.Queue):
    """
    The entry point for the live chat worker process.
    Sets up and runs the asyncio event loop.
    """
    # Each process needs to set up its own project root for imports
    setup_project_root()
    
    # Create and run the asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Create the main runner task
        main_task = loop.create_task(runner(shutdown_event, toggle_event, output_queue))
        # Run until the task is complete
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in livechat_worker.")
    finally:
        # Ensure all tasks are cancelled before closing the loop
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        # Gather all tasks to let them finish their cancellation
        group = asyncio.gather(*tasks, return_exceptions=True)
        loop.run_until_complete(group)
        loop.close()
        logger.info("Asyncio loop closed.")