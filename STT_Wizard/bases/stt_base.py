from abc import ABC, abstractmethod
import numpy as np

class STTBase(ABC):
    """Abstract base class for Speech-to-Text implementations."""

    def __init__(self, model_path: str = None, device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the STTBase.

        Args:
            model_path (str, optional): Path to the STT model. Defaults to None.
            device (str, optional): Device to run the model on ('cpu', 'cuda'). Defaults to "cpu".
            compute_type (str, optional): Compute type for the model (e.g., 'int8', 'float16'). Defaults to "int8".
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Loads the STT model. To be implemented by subclasses."""
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_data: np.ndarray, **kwargs) -> str:
        """
        Transcribes the given audio data.

        Args:
            audio_data (np.ndarray): NumPy array containing the audio data (e.g., float32).
            **kwargs: Additional keyword arguments for transcription (e.g., language, beam_size).

        Returns:
            str: The transcribed text.
        """
        pass

    @abstractmethod
    async def listen_and_transcribe(self, callback, **kwargs):
        """
        Listens for audio input (e.g., from a microphone or stream) and transcribes it.
        This method should handle the audio input and VAD (Voice Activity Detection) if applicable.

        Args:
            callback: An asynchronous callback function to be called with the transcribed text.
            **kwargs: Additional keyword arguments for listening and transcription.
        """
        pass

    async def transcribe_audio_file(self, file_path: str, **kwargs) -> str:
        """
        Transcribes an audio file.
        Subclasses should implement this if they support direct file transcription.
        By default, it raises a NotImplementedError.

        Args:
            file_path (str): Path to the audio file.
            **kwargs: Additional keyword arguments for transcription.

        Returns:
            str: The transcribed text.

        Raises:
            NotImplementedError: If the subclass does not support direct file transcription.
        """
        raise NotImplementedError("Transcribing audio from a file path is not implemented by this STT engine.")

    async def transcribe_and_diarize(self, audio_data: np.ndarray, **kwargs) -> dict:
        """
        Transcribes audio and performs speaker diarization.

        Args:
            audio_data (np.ndarray): NumPy array containing the audio data.
            **kwargs: Additional keyword arguments for transcription and diarization.

        Returns:
            dict: A dictionary containing the transcription and diarization results.
                  Example: {'segments': [{'speaker': 'SPEAKER_01', 'start': 0.5, 'end': 2.3, 'text': 'Hello world'}]}

        Raises:
            NotImplementedError: If the subclass does not support diarization.
        """
        raise NotImplementedError("Diarization is not implemented by this STT engine.")