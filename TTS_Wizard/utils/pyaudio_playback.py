import pyaudio
from .audio_playback_base import AudioPlaybackBase
import logging # Import logging for default logger

class PyAudioPlayback(AudioPlaybackBase):
    """PyAudio implementation for audio playback."""

    def __init__(self, config: dict = None, logger=None):
        super().__init__(config, logger)
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        self._is_paused = False # Internal state for pause/resume

        format_map = {
            'paInt8': pyaudio.paInt8,
            'paInt16': pyaudio.paInt16,
            'paInt24': pyaudio.paInt24,
            'paInt32': pyaudio.paInt32,
            'paFloat32': pyaudio.paFloat32,
            'paUInt8': pyaudio.paUInt8
            # Add other formats as needed
        }

        # Default configuration, can be overridden by config dict
        self.format = config.get('format', 'paInt16') if config else 'paFloat32'
        self.format = format_map.get(self.format, pyaudio.paInt16)
        self.channels = config.get('channels', 1) if config else 1
        self.rate = config.get('rate', 32000) if config else 24000
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
                self._is_paused = False # Ensure pause state is reset

                self.logger.info("PyAudio stream opened.")
            except Exception as e:
                self.logger.error(f"Failed to open PyAudio stream: {e}")
                self.stream = None # Ensure stream is None on failure
                self._is_paused = False # Ensure pause state is reset
                raise # Re-raise the exception so AudioStreamService can handle it

    def write_chunk(self, chunk_data: bytes):
        if self._is_paused:
            self.logger.debug("Attempted to write to a paused PyAudio stream. Chunk ignored.")
            return
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
                self._is_paused = False
        # Terminate PyAudio instance when playback object is no longer needed.
        # This should ideally be called when the PyAudioPlayback instance itself is being destroyed.
        # For now, we'll call it here, but a __del__ or explicit cleanup method might be better.
        # self.pyaudio_instance.terminate() # Moved to a dedicated cleanup method or __del__


    def pause_stream(self):
        if self.stream and self.stream.is_active() and not self._is_paused:
            try:
                self.stream.stop_stream()
                self._is_paused = True
                self.logger.info("PyAudio stream paused.")
            except Exception as e:
                self.logger.error(f"Error pausing PyAudio stream: {e}")
        elif self._is_paused:
            self.logger.info("PyAudio stream is already paused.")
        elif not self.stream: # PyAudio < 0.2.14 might not have is_open
             self.logger.warning("Cannot pause: PyAudio stream is not open.")
        else: # Stream exists but is not active (e.g. already stopped but not by our pause)
            self.logger.info("Cannot pause: PyAudio stream is not currently active (might be already stopped or closed).")


    def resume_stream(self):
        if self.stream and self._is_paused:
            try:
                # Ensure stream is open before trying to start it.
                # PyAudio stream.is_stopped() is true if stop_stream() was called and stream is not closed.
                # PyAudio stream.is_active() is true if start_stream() was called and stream is not closed/stopped.
                # if hasattr(self.stream, 'is_ac') and not self.stream.is_open(): # Check for newer PyAudio
                #     self.logger.warning("Cannot resume: Stream is closed. Needs to be reopened.")
                #     # self._is_paused = False # It's not paused if it's closed
                #     return
                if not self.stream.is_active() and (hasattr(self.stream, 'is_stopped') and self.stream.is_stopped() or not hasattr(self.stream, 'is_stopped')):
                    # If it's stopped (which our pause does) or we can't check is_stopped (older PyAudio)
                    self.stream.start_stream()
                    # self._is_paused = False
                    self.logger.info("PyAudio stream resumed.")
                elif self.stream.is_active(): # Already active, implies not paused by us
                    self.logger.info("Stream is already active, un-pausing.")
                self._is_paused = False
                
            except Exception as e:
                self.logger.error(f"Error resuming PyAudio stream: {e}")
                # If resume fails, stream might be in a bad state. Consider closing.
                self.close_stream()
        elif not self.stream:
            self.logger.warning("Cannot resume: PyAudio stream does not exist.")
        elif not self._is_paused:
            self.logger.info("PyAudio stream is not paused, no need to resume.")


    def stop_and_clear_internal_buffers(self):
        """
        For PyAudio, stopping the stream effectively clears its internal hardware/driver buffers
        as no new data is fed and existing data finishes playing or is cut off.
        This method ensures the stream is stopped. It does not close the stream object itself,
        allowing for a potential resume or explicit close later.
        """
        if self.stream and self.stream.is_active():
            try:
                self.stream.stop_stream()
                # self._is_paused = True # Stopping output can be seen as a form of pause
                self.logger.info("PyAudio stream stopped (output and internal buffers cleared/halted).")
            except Exception as e:
                self.logger.error(f"Error stopping PyAudio stream for buffer clear: {e}")
        elif self.stream and not self.stream.is_active():
            self.logger.info("PyAudio stream already stopped.")
        else:
            self.logger.warning("Cannot stop: PyAudio stream does not exist or is not open.")
        self.stream.start_stream()#reopen stream for audio playback
        


    def is_paused(self) -> bool:
        # Additionally, if the stream object doesn't exist, it can't be paused.
        if not self.stream:
            return False
        return self._is_paused



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