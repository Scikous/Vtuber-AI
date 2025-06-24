class STTCallbacks():
    def __init__(self, logger: logging.Logger, speaker_name: str, speech_queue: asyncio.Queue):
        self.logger = logger
        self.speaker_name = speaker_name
        self.speech_queue = speech_queue

    async def rstt_callback(self, speech_text: str, is_final: bool):
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