from abc import ABC, abstractmethod

class AudioPlaybackBase(ABC):
    """Abstract base class for audio playback implementations."""

    @abstractmethod
    def __init__(self, config: dict = None, logger=None):
        """Initialize the audio playback backend.

        Args:
            config (dict, optional): Configuration specific to the backend.
                                     Example: {'format': pyaudio.paInt16, 'channels': 1, 
                                              'rate': 32000, 'chunk_size': 1024}.
            logger (logging.Logger, optional): Logger instance.
        """
        pass

    @abstractmethod
    def open_stream(self):
        """Open or re-open the audio output stream."""
        pass

    @abstractmethod
    def write_chunk(self, chunk_data: bytes):
        """Write a chunk of audio data to the stream.

        Args:
            chunk_data (bytes): The audio data to play.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if the audio stream is currently active.

        Returns:
            bool: True if the stream is active, False otherwise.
        """
        pass

    @abstractmethod
    def close_stream(self):
        """Stop and close the audio stream and release resources."""
        pass

    @abstractmethod
    def pause_stream(self):
        """Pause the audio stream if it is currently playing."""
        pass

    @abstractmethod
    def resume_stream(self):
        """Resume the audio stream if it is currently paused."""
        pass

    @abstractmethod
    def stop_and_clear_internal_buffers(self):
        """
        Stop the audio stream immediately and clear any internal hardware/driver buffers.
        This does not necessarily close the stream, but ensures playback ceases and
        pending data in low-level buffers is discarded.
        The service using this backend is responsible for clearing its own software buffers.
        """
        pass

    @abstractmethod
    def is_paused(self) -> bool:
        """Check if the audio stream is currently paused.

        Returns:
            bool: True if the stream is paused, False otherwise.
        """
        pass


    def __enter__(self):
        self.open_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_stream()