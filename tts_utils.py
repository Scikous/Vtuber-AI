import logging
import asyncio

class TTSController:
    def __init__(self, logger=None):
        self._tts_paused = False
        self._tts_stopped = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        self.logger = logger

    async def pause_tts(self):
        if not self._tts_paused:
            self._tts_paused = True
            self._pause_event.clear() # Signal to pause
            if self.logger:
                self.logger.info("TTS explicitly paused.")
        elif self.logger:
            self.logger.info("TTS is already paused.")

    async def resume_tts(self):
        if self._tts_paused:
            self._tts_paused = False
            self._pause_event.set() # Signal to resume
            if self.logger:
                self.logger.info("TTS explicitly resumed.")
        elif self.logger:
            self.logger.info("TTS is not paused, no need to resume.")

    async def stop_tts(self):
        self._tts_stopped = True
        self._pause_event.set() # Unblock any waiting
        if self.logger:
            self.logger.info("TTS explicitly stopped.")

    async def wait_if_paused(self):
        await self._pause_event.wait()

    def is_paused(self):
        return self._tts_paused

    def is_stopped(self):
        return self._tts_stopped

def prepare_tts_params(text_to_speak, text_lang="en", ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav", prompt_text="", prompt_lang="en", streaming_mode=False, media_type="wav", logger=None):
    """
    Prepares the dictionary of parameters for the TTS service.
    All parameters are now passed directly to the function.
    """
    return {
        "text": text_to_speak,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "streaming_mode": streaming_mode,
        "media_type": media_type,
    }