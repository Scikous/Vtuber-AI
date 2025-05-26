import pyaudio
from .audio_playback_base import AudioPlaybackBase
import logging # Import logging for default logger

class PyAudioPlayback(AudioPlaybackBase):
    """PyAudio implementation for audio playback."""

    def __init__(self, config: dict = None, logger=None):
        super().__init__(config, logger)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
        # Default configuration, can be overridden by config dict
        self.format = config.get('format', pyaudio.paInt16) if config else pyaudio.paInt16
        self.channels = config.get('channels', 1) if config else 1
        self.rate = config.get('rate', 32000) if config else 32000
        self.chunk_size = config.get('chunk_size', 1024) if config else 1024
        
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.info("PyAudioPlayback initialized.")

    def open_stream(self):
        if not self.stream or not self.is_active(): # Check is_active for robustness
            try:
                self.stream = self.pyaudio_instance.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )
                self.logger.info("PyAudio stream opened.")
            except Exception as e:
                self.logger.error(f"Failed to open PyAudio stream: {e}")
                self.stream = None # Ensure stream is None on failure
                raise # Re-raise the exception so AudioStreamService can handle it

    def write_chunk(self, chunk_data: bytes):
        if self.stream and self.is_active():
            try:
                self.stream.write(chunk_data)
            except Exception as e:
                self.logger.error(f"Error writing to PyAudio stream: {e}")
                # Optionally, try to reopen stream or handle error
                self.close_stream() # Close potentially broken stream
                raise # Re-raise to signal failure to the caller
        else:
            self.logger.warning("Attempted to write to a closed or inactive PyAudio stream.")
            # raise IOError("PyAudio stream is not open or active.") # Or raise an error

    def is_active(self) -> bool:
        return self.stream is not None and self.stream.is_active()

    def close_stream(self):
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.logger.info("PyAudio stream closed.")
            except Exception as e:
                self.logger.error(f"Error closing PyAudio stream: {e}")
            finally:
                self.stream = None
        # Terminate PyAudio instance when playback object is no longer needed.
        # This should ideally be called when the PyAudioPlayback instance itself is being destroyed.
        # For now, we'll call it here, but a __del__ or explicit cleanup method might be better.
        # self.pyaudio_instance.terminate() # Moved to a dedicated cleanup method or __del__

    def cleanup(self):
        """Explicitly cleans up PyAudio resources."""
        self.close_stream() # Ensure stream is closed first
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.logger.info("PyAudio instance terminated.")
            except Exception as e:
                self.logger.error(f"Error terminating PyAudio instance: {e}")
            finally:
                self.pyaudio_instance = None

    def __del__(self):
        # This is a fallback. Explicit cleanup is preferred.
        self.logger.info("PyAudioPlayback __del__ called. Cleaning up...")
        self.cleanup()