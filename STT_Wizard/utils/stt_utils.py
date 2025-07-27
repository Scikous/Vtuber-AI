
from .config import load_config
import numpy as np
import math
import logging

import asyncio
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Awaitable, Any

class STTBase(ABC):
    """
    Abstract Base Class for Speech-to-Text (STT) implementations.

    This class defines the standard interface for all STT engines, ensuring they
    can be used interchangeably. It handles model initialization and defines
    the core transcription methods that subclasses must implement.
    """

    def __init__(self, model_path: str = None, language: str = None, device: str = None, compute_type: str = None, **stt_settings):
        """
        Initializes the STT engine.

        Args:
            model_path (str, optional): Path or identifier for the STT model.
            language: Language code for transcription (e.g., 'en')
            device (str, optional): The device to run the model on (e.g., 'cpu', 'cuda').
            compute_type (str, optional): The computation type for the model (e.g., 'int8', 'float16').
            **stt_settings: Additional keyword arguments for the specific implementation. Namely STT settings
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        # self._load_model() # Model loading is now explicit

    @abstractmethod
    def _load_model(self):
        """
        Loads the specific STT model into memory.

        Subclasses must implement this method to handle the loading of their
        respective models and prepare them for transcription. This method
        is called automatically during initialization.
        """
        raise NotImplementedError("Subclasses must implement the _load_model method.")

    @abstractmethod
    async def transcribe_audio(self, audio_data: np.ndarray, **kwargs) -> str:
        """
        Transcribes a given audio segment.

        This method takes a complete audio segment as a NumPy array and returns
        the transcribed text.

        Args:
            audio_data (np.ndarray): The audio data to be transcribed.
            **kwargs: Additional implementation-specific parameters for transcription.

        Returns:
            str: The resulting transcription text.
        """
        raise NotImplementedError("Subclasses must implement the transcribe_audio method.")

    @abstractmethod #could be async as well -- mayhaps
    def listen_and_transcribe(self, callback: Callable[[str, bool], Awaitable[None]], **kwargs):
        """
        Listens to an audio input stream and provides real-time transcriptions.

        This method should handle capturing audio from a microphone, detecting speech,
        and calling the provided callback function with transcription results.

        Args:
            callback: An awaitable callback function to be invoked with transcription results.
                      It receives two arguments:
                      - text (str): The transcribed text.
                      - is_final (bool): True if the transcription segment is final.
            **kwargs: Additional parameters, typically including:
                      - stop_event (threading.Event): An event to signal when to stop listening.
                      - loop (asyncio.AbstractEventLoop): The event loop to run the callback in.
                      - device_index (int, optional): The index of the audio input device.
        """
        raise NotImplementedError("Subclasses must implement the listen_and_transcribe method.")

    def list_available_input_devices(self) -> list:
        """
        Provides a list of available audio input devices.

        This is a concrete utility method that can be used by any subclass.

        Returns:
            list: A list of available input devices, as provided by the helper function.
        """
        try:
            return self.list_available_input_devices()
        except Exception as e:
            print(f"Could not list audio devices. Is 'sounddevice' installed and working? Error: {e}")
            return []



def calculate_audio_energy_rms(audio_chunk: np.ndarray) -> float:
    """
    Calculates the Root Mean Square (RMS) energy of an audio chunk.
    Assumes audio_chunk is a NumPy array of audio samples (e.g., float32).
    Returns 0.0 if the chunk is empty or contains non-finite values.
    """
    if audio_chunk is None or audio_chunk.size == 0:
        return 0.0
    
    # Ensure the data is float to avoid overflow with int types before squaring
    audio_chunk_float = audio_chunk.astype(np.float64) if audio_chunk.dtype != np.float64 else audio_chunk
    
    # Check for non-finite values which would cause issues
    if not np.all(np.isfinite(audio_chunk_float)) :
        print("Warning: Non-finite values detected in audio chunk for RMS calculation.")
        # One option is to replace non-finite with 0, or simply return 0 for the chunk
        # audio_chunk_float = np.nan_to_num(audio_chunk_float)
        return 0.0 # Or handle as appropriate

    if audio_chunk_float.size == 0: # Re-check after potential nan_to_num if that path was taken
        return 0.0

    rms = np.sqrt(np.mean(np.square(audio_chunk_float)))
    return float(rms) if np.isfinite(rms) else 0.0


def calculate_dbfs(rms_energy: float, max_possible_value: float = 1.0) -> float:
    """
    Calculates dBFS (decibels relative to full scale) from RMS energy.
    Assumes RMS is normalized (e.g., for float32, max_possible_value is 1.0).
    Returns -inf if rms_energy is zero or negative.
    """
    if rms_energy <= 0:
        return -float('inf')
    if max_possible_value <= 0:
        # Avoid log of zero or negative if max_possible_value is bad
        return -float('inf') 
    
    # Ensure rms_energy does not exceed max_possible_value to avoid positive dBFS if not desired
    # This depends on whether the input rms_energy can be clipped or not.
    # For now, we assume rms_energy could be > max_possible_value if source is too loud.
    # rms_energy = min(rms_energy, max_possible_value)

    dbfs = 20 * math.log10(rms_energy / max_possible_value)
    return dbfs


def count_words(text: str) -> int:
    """
    Counts the number of words in a given text string.
    Words are assumed to be separated by whitespace.
    """
    if not text or not text.strip():
        return 0
    return len(text.split())

def create_stt_callback(speech_queue, speaker_name="User", logger=None):
    """Create a standardized STT callback function for multiprocessing workers.
    
    Args:
        speech_queue: Queue to put recognized speech into
        speaker_name: Name of the speaker (default: "User")
        logger: Logger instance for debugging
        
    Returns:
        Callback function that can be used with speech_to_text
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def stt_callback(speech):
        """Callback function for STT results."""
        try:
            # Filter out the common false positive
            if speech and speech.strip().lower() != "thank you.":
                message = f"{speaker_name}: {speech.strip()}"
                
                # Put message in queue (non-blocking)
                try:
                    speech_queue.put_nowait(message)
                    logger.debug(f"STT captured: {speech.strip()}")
                except:
                    # Queue is full, try to make space by removing oldest item
                    try:
                        speech_queue.get_nowait()  # Remove oldest
                        speech_queue.put_nowait(message)  # Add new
                        logger.debug(f"STT captured (queue was full): {speech.strip()}")
                    except:
                        logger.warning(f"STT queue full, dropped message: {speech.strip()}")
        except Exception as e:
            logger.error(f"Error in STT callback: {e}")
    
    return stt_callback

# Example usage (can be removed or kept for testing):
if __name__ == '__main__':
    # Test audio energy
    silence = np.zeros(1024, dtype=np.float32)
    quiet_sound = np.random.uniform(-0.1, 0.1, 1024).astype(np.float32)
    loud_sound = np.random.uniform(-0.8, 0.8, 1024).astype(np.float32)
    full_scale_sine = np.sin(np.linspace(0, 2 * np.pi * 10, 1024)).astype(np.float32) # Approx 0.707 RMS

    print(f"Energy of silence: {calculate_audio_energy_rms(silence):.4f}, dBFS: {calculate_dbfs(calculate_audio_energy_rms(silence)):.2f}")
    print(f"Energy of quiet sound: {calculate_audio_energy_rms(quiet_sound):.4f}, dBFS: {calculate_dbfs(calculate_audio_energy_rms(quiet_sound)):.2f}")
    print(f"Energy of loud sound: {calculate_audio_energy_rms(loud_sound):.4f}, dBFS: {calculate_dbfs(calculate_audio_energy_rms(loud_sound)):.2f}")
    print(f"Energy of full_scale_sine: {calculate_audio_energy_rms(full_scale_sine):.4f}, dBFS: {calculate_dbfs(calculate_audio_energy_rms(full_scale_sine)):.2f}")

    non_finite_sound = np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float32)
    print(f"Energy of non_finite_sound: {calculate_audio_energy_rms(non_finite_sound):.4f}")
    
    empty_sound = np.array([], dtype=np.float32)
    print(f"Energy of empty_sound: {calculate_audio_energy_rms(empty_sound):.4f}")


    # Test word count
    text1 = "Hello world, this is a test."
    text2 = "OneWord"
    text3 = "   Leading and trailing spaces   "
    text4 = ""
    text5 = "    " # Only spaces
    
    print(f"Words in '{text1}': {count_words(text1)}") # Expected: 6
    print(f"Words in '{text2}': {count_words(text2)}") # Expected: 1
    print(f"Words in '{text3}': {count_words(text3)}") # Expected: 5
    print(f"Words in '{text4}': {count_words(text4)}") # Expected: 0
    print(f"Words in '{text5}': {count_words(text5)}") # Expected: 0
