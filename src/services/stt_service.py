"""
STT (Speech-to-Text) Service Module for Vtuber-AI
"""
import asyncio
import threading
import traceback
from .base_service import BaseService
from STT_Wizard.STT import WhisperSTT # STT CLASS
from STT_Wizard.utils.callbacks import STTCallbacks
class STTService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.speech_queue = self.queues.get("speech_queue")
        
        stt_settings = self.config.get("stt_settings", {})
        self.speaker_name = stt_settings.get("speaker_name", "User")
        self.model_size = stt_settings.get("MODEL_SIZE", "large-v3")
        self.language = stt_settings.get("LANGUAGE", "en")
        self.device = stt_settings.get("DEVICE", "cuda")
        self.compute_type = stt_settings.get("COMPUTE_TYPE", "int8_float16")
        self.device_index = stt_settings.get("DEVICE_INDEX", 0)

        self.stt_is_listening_event = self.shared_resources.get(
            "stt_is_listening_event", threading.Event()
            )
        self.stt_can_finish_event = self.shared_resources.get(
            "stt_can_finish_event", threading.Event()
            )
        
        self.STTCLASS = WhisperSTT(self.model_size, self.language, self.device, self.compute_type, self.stt_is_listening_event, self.stt_can_finish_event, **stt_settings)
        # Service-specific state for managing the thread
        self._stt_thread = None
        self._stop_thread_event = threading.Event()


        if self.logger:
            # Create the callbacks instance
            stt_callbacks = STTCallbacks(self.logger, self.speaker_name, self.speech_queue)
            self._stt_callback = stt_callbacks.rstt_callback #real-time stt_callback
            
            self.logger.info(f"STTService initialized for speaker '{self.speaker_name}'.")

    async def run_worker(self):
        """
        The main worker coroutine for the STT service.
        This method is started as a task by the BaseService's `start()` method.
        """
        self.logger.info("STT worker starting...")
        loop = asyncio.get_running_loop()
        self._stop_thread_event.clear()
        
        try:
            # --- Setup Phase ---
            self.logger.info("Starting recognizer thread.")
            self._stt_thread = threading.Thread(
                target=self.STTCLASS.listen_and_transcribe,
                args=(self._stt_callback, self._stop_thread_event, loop, self.device_index)
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