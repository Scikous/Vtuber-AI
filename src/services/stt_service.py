"""
STT (Speech-to-Text) Service Module for Vtuber-AI
"""
import asyncio
from .base_service import BaseService
from STT_Wizard.STT import speech_to_text # Import for STT functionality

class STTService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        
        # Get speech queue from base class (already available through self.queues)
        self.speech_queue = self.queues.get("speech_queue") # Output queue for STT results
        
        # Get speaker name from config instead of shared_resources
        self.stt_settings = self.config.get("stt_settings", {}) if self.config else {}
        self.speaker_name = self.stt_settings.get("speaker_name", "User") if self.stt_settings else "User"

        # Shared state events (to be managed/set externally)
        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event()) # Stops current dialogue playback
        self.is_audio_streaming_event = shared_resources.get("is_audio_streaming_event", asyncio.Event()) # Stops current dialogue playback
        
        if self.logger:
            self.logger.info(f"STTService initialized with speaker name: {self.speaker_name}")

    async def run_worker(self):
        """Main logic for the STT service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")
        try:
            async def stt_callback(speech):
                # Current STT system recognizes no sound as 'Thank you.' for reasons unknown
                if speech and speech.strip().lower() != "thank you.": # Made comparison case-insensitive
                    if self.speech_queue:
                        await self.speech_queue.put(f"{self.speaker_name}: {speech.strip()}")
                        if self.logger:
                            self.logger.debug(f"STT captured: {speech.strip()}")
                    elif self.logger:
                        self.logger.warning("Speech queue not available in STTService.")
                # Optional: Log ignored "Thank you." messages if needed for debugging
                # elif self.logger and speech and speech.strip().lower() == "thank you.":
                #     self.logger.debug("STT ignored 'Thank you.'")

            while True:
                await speech_to_text(stt_callback, self.terminate_current_dialogue_event, self.is_audio_streaming_event) # speech_to_text is a blocking call in a loop, ensure it yields or is async
                await asyncio.sleep(0.1) # Brief sleep to yield control
        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")
