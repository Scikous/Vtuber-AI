"""
STT (Speech-to-Text) Service Module for Vtuber-AI
"""
import asyncio
import threading
import traceback
from .base_service import BaseService
from STT_Wizard.STT import recognize_speech_stream # We only need the core function

class STTService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.speech_queue = self.queues.get("speech_queue")
        
        stt_settings = self.config.get("stt_settings", {})
        self.speaker_name = stt_settings.get("speaker_name", "User")
        self.device_index = stt_settings.get("device_index")

        # Service-specific state for managing the thread
        self._stt_thread = None
        self._stop_thread_event = threading.Event()

        if self.logger:
            self.logger.info(f"STTService initialized for speaker '{self.speaker_name}'.")

    async def _stt_callback(self, speech_text: str, is_final: bool):
        """Async callback to handle transcribed text from the STT thread."""
        # This callback is thread-safe because it's scheduled on the main event loop.
        
        # Filter out the erroneous "Thank you." transcriptions
        if speech_text and speech_text.strip().lower() != "thank you.":
            try:
                # You can use the 'is_final' flag for more nuanced logic if needed
                if is_final:
                    self.logger.info(f"Final STT transcription: '{speech_text.strip()}'")
                    await self.speech_queue.put(f"{self.speaker_name}: {speech_text.strip()}")
                else:
                    self.logger.debug(f"Interim STT transcription: '{speech_text.strip()}'")
                
                
            except asyncio.QueueFull:
                self.logger.warning("Speech queue is full. Discarding transcription.")
            except Exception as e:
                self.logger.error(f"Error in STT callback: {e}", exc_info=True)

    async def run_worker(self):
        """
        The main worker coroutine for the STT service.
        This method is started as a task by the BaseService's `start()` method.
        """
        self.logger.info("STT worker starting...")
        loop = asyncio.get_running_loop()
        self._stop_thread_event.clear()

        # A wrapper to adapt the callback signature and schedule it on the event loop
        def callback_adapter(text, is_final):
            asyncio.run_coroutine_threadsafe(self._stt_callback(text, is_final), loop)

        try:
            # --- Setup Phase ---
            self.logger.info("Starting recognizer thread.")
            self._stt_thread = threading.Thread(
                target=recognize_speech_stream,
                args=(callback_adapter, self._stop_thread_event, loop, self.device_index)
            )
            self._stt_thread.start()

            # --- Running Phase ---
            # The worker will stay in this state until it's cancelled by the service's stop() method.
            await asyncio.Future()

        except asyncio.CancelledError:
            self.logger.info("STT worker cancellation requested.")
            # The cancellation is expected; the 'finally' block handles the shutdown.

        except Exception as e:
            self.logger.error(f"A critical error occurred in the STT worker: {e}", exc_info=True)

        finally:
            # --- Cleanup Phase ---
            self.logger.info("Cleaning up STT worker...")
            if self._stt_thread and self._stt_thread.is_alive():
                # 1. Signal the thread to stop its internal loop.
                self._stop_thread_event.set()
                self.logger.info("Waiting for STT thread to join...")
                
                # 2. Wait for the thread to finish in a non-blocking way.
                try:
                    await loop.run_in_executor(None, self._stt_thread.join, 2.0) # 2-second timeout
                    self.logger.info("STT thread joined successfully.")
                except Exception as e:
                    self.logger.error(f"Error while joining STT thread: {e}")
            
            self._stt_thread = None
            self.logger.info("STT worker has stopped.")