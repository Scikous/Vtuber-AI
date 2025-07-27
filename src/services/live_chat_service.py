"""
Live Chat Service Module for Vtuber-AI
Handles interactions with live chat platforms (e.g., YouTube, Twitch).
"""
import asyncio
import multiprocessing # Added for multiprocessing.queues.Empty
from .base_service import BaseService
from Livechat_Wizard.livechat import LiveChatController

class LiveChatService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        # Input from the live chat platform (e.g., YouTube API)
        self.live_chat_controller = LiveChatController.create()
        
        # Get live chat queue from base class (already available through self.queues)
        self.live_chat_input_queue = self.queues.get("live_chat_queue")
        
        # Event flag for immediate fetch trigger
        self.immediate_livechat_fetch_event = shared_resources.get("immediate_livechat_fetch_event", asyncio.Event()) if shared_resources else asyncio.Event()
        
        # Get fetch interval from config (default 60 seconds)
        self.livechat_settings = self.config.get("livechat_settings", {}) if self.config else {}
        self.fetch_interval = self.livechat_settings.get("live_chat_fetch_interval", 60) if self.livechat_settings else 60
        
        if self.logger:
            self.logger.info(f"LiveChatService initialized with fetch interval: {self.fetch_interval}s")

    def trigger_immediate_fetch(self):
        """Trigger an immediate fetch by setting the event flag."""
        self.immediate_livechat_fetch_event.set()
        if self.logger:
            self.logger.debug("Immediate fetch triggered")

    def set_fetch_interval(self, interval_seconds):
        """Set the fetch interval in seconds."""
        self.fetch_interval = interval_seconds
        if self.logger:
            self.logger.info(f"Fetch interval set to {interval_seconds} seconds")

    async def run_worker(self):
        """Main logic for the Live Chat service worker.
           Retrieves messages from a multiprocessing queue (filled by a separate live chat process)
           and puts them into an asyncio queue for the DialogueService.
        """
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")

        if not self.live_chat_input_queue:
            if self.logger:
                self.logger.error("Live chat input queue (for DialogueService) not available. Stopping worker.")
            return

        try:
            while True:
                try:
                    message, context_messages = await self.live_chat_controller.fetch_chat_message()

                    if message:
                        if self.logger:
                            self.logger.info(f"Message received: {message}, Messages for Context: {context_messages}")
                        # Put the message onto the asyncio queue for DialogueService
                        if not self.live_chat_input_queue.full():
                            await self.live_chat_input_queue.put_nowait((message, context_messages))
                            if self.logger:
                                self.logger.debug(f"LiveChatService worker put to live_chat_input_queue: {message}")
                        else:
                            if self.logger:
                                self.logger.warning("Live chat input queue is full. Message dropped.")
                    
                    # Wait for either timeout OR immediate fetch flag
                    await self._wait_for_next_fetch()

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

    async def _wait_for_next_fetch(self):
        """Wait for either the fetch interval timeout OR immediate fetch flag."""
        try:
            # Wait for the immediate fetch event with a timeout of fetch_interval seconds
            await asyncio.wait_for(self.immediate_livechat_fetch_event.wait(), timeout=self.fetch_interval)
            # If we get here, the event was set (immediate fetch requested)
            if self.logger:
                self.logger.debug("Immediate fetch event triggered, proceeding with fetch")
        except asyncio.TimeoutError:
            # Timeout occurred, proceed with normal fetch
            if self.logger:
                self.logger.debug(f"Fetch interval of {self.fetch_interval}s elapsed, proceeding with fetch")
        finally:
            # Always clear the event for next iteration
            self.immediate_livechat_fetch_event.clear()