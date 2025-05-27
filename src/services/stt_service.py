"""
STT (Speech-to-Text) Service Module for Vtuber-AI
"""
import asyncio
from .base_service import BaseService
from STT_Wizard.STT import speech_to_text # Import for STT functionality

class STTService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.speech_queue = self.queues.get("speech_queue") # Output queue for STT results
        self.speaker_name = self.shared_resources.get("speaker_name", "User") # Get speaker name from shared resources

        # Shared state events (to be managed/set externally)
        self.user_speaking_pause_event = shared_resources.get("user_speaking_pause_event", asyncio.Event()) # Pauses playback when user speaks
        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event()) # Stops current dialogue playback

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
                await speech_to_text(stt_callback, self.user_speaking_pause_event, self.terminate_current_dialogue_event) # speech_to_text is a blocking call in a loop, ensure it yields or is async
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
