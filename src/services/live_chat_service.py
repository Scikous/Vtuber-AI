"""
Live Chat Service Module for Vtuber-AI
Handles interactions with live chat platforms (e.g., YouTube, Twitch).
"""
import asyncio
import multiprocessing # Added for multiprocessing.queues.Empty
from .base_service import BaseService

class LiveChatService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        # Queues for communication
        # Input from the live chat platform (e.g., YouTube API)
        self.mp_live_chat_message_queue = self.queues.get("mp_live_chat_message_queue")
        # Output to the Dialogue service (which consumes from live_chat_queue)
        self.live_chat_input_queue = self.queues.get("live_chat_queue") 

    async def run_worker(self):
        """Main logic for the Live Chat service worker.
           Retrieves messages from a multiprocessing queue (filled by a separate live chat process)
           and puts them into an asyncio queue for the DialogueService.
        """
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")

        if not self.mp_live_chat_message_queue:
            if self.logger:
                self.logger.error("Multiprocessing live chat message queue not available. Stopping worker.")
            return
        if not self.live_chat_input_queue:
            if self.logger:
                self.logger.error("Live chat input queue (for DialogueService) not available. Stopping worker.")
            return

        try:
            while True:
                try:
                    # Get message from the multiprocessing queue (filled by live_chat_process)
                    # This needs to be non-blocking or handled carefully in an async context.
                    # Using get_nowait() and handling the Empty exception is one way.
                    message = self.mp_live_chat_message_queue.get_nowait()
                    
                    if message is None: # Signal that live chat source (e.g. LiveChatController) is not available or stopped
                        if self.logger:
                            self.logger.info("Received None from mp_live_chat_message_queue, live chat source might be unavailable or shutting down.")
                        # Depending on desired behavior, could break or continue to allow restart
                        await asyncio.sleep(1) # Wait a bit before checking again
                        continue

                    if self.logger:
                        self.logger.debug(f"LiveChatService worker received from mp_queue: {message}")

                    # Put the message onto the asyncio queue for DialogueService
                    if not self.live_chat_input_queue.full():
                        await self.live_chat_input_queue.put(message)
                        if self.logger:
                            self.logger.debug(f"LiveChatService worker put to live_chat_input_queue: {message}")
                    else:
                        if self.logger:
                            self.logger.warning("Live chat input queue is full. Message dropped.")
                
                except multiprocessing.queues.Empty: # Specific exception for MPQueue
                    # No message available in mp_live_chat_message_queue, wait a bit
                    await asyncio.sleep(0.1) # Adjust sleep time as needed
                    continue
                except AttributeError as e:
                    if self.logger:
                        self.logger.error(f"AttributeError in LiveChatService worker (often queue not init): {e}")
                    await asyncio.sleep(1)
                    continue # Or break, depending on recovery strategy
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Unexpected error in LiveChatService worker: {e}", exc_info=True)
                    await asyncio.sleep(1) # Avoid rapid error looping

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")