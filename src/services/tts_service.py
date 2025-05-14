"""
TTS (Text-to-Speech) Service Module for Vtuber-AI
"""
import asyncio
import os
from .base_service import BaseService
from TTS_Wizard.GPT_Test.tts_exp import send_tts_request # Assuming this is the correct import
# If run_playback_thread needs to be managed by this service, it would be imported here too.
# from voiceAI.GPT_Test.tts_exp import run_playback_thread

class TTSService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.llm_output_queue = self.queues.get("llm_output_queue") # Input queue for TTS
        self.project_root = self.shared_resources.get("project_root")
        if not self.project_root:
            if self.logger:
                self.logger.error("Project root not found in shared resources for TTSService.")
            # Fallback or raise error, for now, assume it's present
            self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.tts_module_path = os.path.join(self.project_root, "voiceAI", "GPT_Test")
        
        # Placeholder for playback thread management if this service handles it
        # self.playback_thread = None
        # self.stop_event = asyncio.Event() # Or threading.Event if run_playback_thread is not async

    async def run_worker(self):
        """Main logic for the TTS service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")
        try:
            if not self.llm_output_queue:
                if self.logger:
                    self.logger.error("LLM output queue not found in TTSService.")
                return # Cannot operate without the input queue

            while True:
                text_to_speak = await self.llm_output_queue.get()
                if text_to_speak is None: # Sentinel for stopping the service gracefully
                    if self.logger:
                        self.logger.info(f"{self.__class__.__name__} received stop sentinel.")
                    break
                
                if self.logger:
                    self.logger.debug(f"TTS service received: {text_to_speak[:50]}...")

                original_cwd = None
                try:
                    if not os.path.exists(self.tts_module_path):
                        if self.logger:
                            self.logger.error(f"TTS module path does not exist: {self.tts_module_path}")
                        await asyncio.sleep(1) # Avoid busy loop if path is wrong
                        continue
                    
                    original_cwd = os.getcwd()
                    os.chdir(self.tts_module_path)
                    if self.logger:
                        self.logger.debug(f"Changed CWD to {self.tts_module_path} for TTS request.")
                    
                    # Assuming send_tts_request is an async function or can be awaited
                    # If it's blocking, it should be run in an executor.
                    await send_tts_request(text_to_speak)
                    
                    if self.logger:
                        self.logger.debug(f"TTS request sent for: {text_to_speak[:50]}...")
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during TTS processing in {self.tts_module_path}: {e}", exc_info=True)
                finally:
                    if original_cwd:
                        os.chdir(original_cwd)
                        if self.logger:
                            self.logger.debug(f"Restored CWD to {original_cwd}.")
                
                await asyncio.sleep(0.1) # Yield control and prevent tight loop
        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")

    # If TTSService manages the playback thread, start/stop methods would be overridden:
    # async def start(self):
    #     # Start the playback thread if not already running
    #     # Example: 
    #     # if not self.playback_thread or not self.playback_thread.is_alive():
    #     #     self.stop_event.clear()
    #     #     # Assuming run_playback_thread takes a stop_event and is designed to be run in a thread
    #     #     self.playback_thread = threading.Thread(target=run_playback_thread, args=(self.stop_event,), daemon=True)
    #     #     self.playback_thread.start()
    #     #     if self.logger:
    #     #         self.logger.info("TTS playback thread started.")
    #     await super().start() # Start the run_worker task

    # async def stop(self):
    #     # Stop the run_worker task first
    #     await super().stop()
    #     # Then stop the playback thread
    #     # Example:
    #     # if self.playback_thread and self.playback_thread.is_alive():
    #     #     if self.logger:
    #     #         self.logger.info("Stopping TTS playback thread...")
    #     #     self.stop_event.set()
    #     #     self.playback_thread.join(timeout=5) # Wait for thread to finish
    #     #     if self.playback_thread.is_alive():
    #     #         if self.logger:
    #     #             self.logger.warning("TTS playback thread did not stop in time.")
    #     #     else:
    #     #         if self.logger:
    #     #             self.logger.info("TTS playback thread stopped.")
    #     # self.playback_thread = None
        pass