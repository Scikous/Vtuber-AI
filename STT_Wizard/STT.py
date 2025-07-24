import asyncio
import time
import traceback

# Attempt to import necessary libraries for the new STT solution
try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import numpy as np
    import webrtcvad
    from collections import deque
    import threading
    import io
    import wave
    from .utils.config import load_config
    
except ImportError as e:
    print(f"Error importing necessary libraries for STT: {e}")
    print("Please ensure 'faster-whisper', 'sounddevice', 'numpy', 'webrtcvad-wheels' (or 'webrtcvad') are installed.")
    # Fallback or raise error if critical components are missing
    # For now, we'll let it proceed and fail later if these are used without being imported.
    WhisperModel = None # Placeholder to avoid immediate crash if script is imported but not run

from .utils.stt_utils import calculate_audio_energy_rms, calculate_dbfs, count_words, STTBase

class WhisperSTT(STTBase):
    """Whisper-based Speech-to-Text implementation using faster-whisper."""
    
    def __init__(self, model_size: str = None, language: str = None, device: str = None, compute_type: str = None, stt_is_listening_event: threading.Event = None, stt_can_finish_event: threading.Event = None, **stt_settings):
        """Initialize WhisperSTT with configuration.
        
        Args:
            model_size: Whisper model size -- Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo, "distil-whisper/distil-large-v3.5-ct2"
            language: Language code for transcription (e.g., 'en')
            device: Device to run on ('cpu', 'cuda')
            compute_type: Compute type ('int8', 'float16', 'int8_float16', etc.)
            **stt_settings: Additional keyword arguments for the specific implementation. Namely STT settings
        
        """
        # Load configuration
        if not stt_settings:
            self.config = load_config()
        else:
            self.config = stt_settings
        # Set parameters with fallbacks to config
        self.model_size = model_size or self.config.get("MODEL_SIZE", "large-v3")
        self.language = language or self.config.get("LANGUAGE", "en")
        self.beam_size = self.config.get("BEAM_SIZE", 5)
        
        # Determine device and compute type
        if device is None or compute_type is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = device or "cuda"
                    self.compute_type = compute_type or "int8_float16"
                    print(f"PyTorch found. Using device: {self.device} with compute type: {self.compute_type}")
                else:
                    self.device = device or "cpu"
                    self.compute_type = compute_type or "int8"
                    print(f"CUDA not available. Using device: {self.device} with compute type: {self.compute_type}")
            except ImportError:
                self.device = device or "cpu"
                self.compute_type = compute_type or "int8"
                print("PyTorch not found. Defaulting to CPU for faster-whisper.")
        else:
            self.device = device
            self.compute_type = compute_type
        
        # Audio parameters
        self.sample_rate = self.config.get("SAMPLE_RATE", 16000)
        self.channels = self.config.get("CHANNELS", 1)
        self.audio_dtype = self.config.get("AUDIO_DTYPE", "float32")
        self.frame_duration_ms = self.config.get("FRAME_DURATION_MS", 10)
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.vad_aggressiveness = self.config.get("VAD_AGGRESSIVENESS", 3)
        
        # Speech detection parameters
        self.silence_duration_s_for_finalize = self.config.get("SILENCE_DURATION_S_FOR_FINALIZE", 0.5)
        self.proactive_pause_s = self.config.get("PROACTIVE_PAUSE_S", 0.05)
        self.min_chunk_s_for_interim = self.config.get("MIN_CHUNK_S_FOR_INTERIM", 0.1)
        self.energy_threshold_dbfs = self.config.get("ENERGY_THRESHOLD_DBFS", -40)
        
        # Audio window configuration
        self.audio_window_s = self.config.get("AUDIO_WINDOW_S", 6)
        self.audio_window_frames = int(self.audio_window_s * self.sample_rate)
        
        self.stt_is_listening_event = stt_is_listening_event
        self.stt_can_finish_event = stt_can_finish_event
        # Initialize the parent class
        # The model is now loaded explicitly, not in the constructor.
        self.model = None
        # super().__init__(model_path=self.model_size, device=self.device, compute_type=self.compute_type)
    
    def _load_model(self):
        """Load the Whisper model."""
        if self.model:
            print("Model already loaded.")
            return

        if not WhisperModel:
            print("WhisperModel could not be imported. STT will not function.")
            self.model = None
            return
            
        print(f"Loading faster-whisper model: {self.model_size}...")
        try:
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("faster-whisper model loaded successfully.")
            # Warmup the model
            self._warmup_model()
        except Exception as e:
            print(f"Error loading faster-whisper model: {e}")
            self.model = None
    
    def _warmup_model(self):
        """Warm up the STT model by running a dummy transcription."""
        if not self.model:
            print("Warmup skipped: Whisper model is not available.")
            return

        print("Warming up the STT model...")
        start_time = time.monotonic()
        
        # Create a short, silent audio array
        silent_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        
        try:
            # Use simple parameters for the warmup
            _, _ = self.model.transcribe(silent_audio, beam_size=1, language=self.language)
            
            end_time = time.monotonic()
            print(f"STT model warmed up in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred during STT model warmup: {e}")
    
    async def transcribe_audio(self, audio_data: np.ndarray, **kwargs) -> str:
        """Transcribe audio data using Whisper.
        
        Args:
            audio_data: NumPy array containing audio data
            **kwargs: Additional transcription parameters
            
        Returns:
            Transcribed text
        """
        if not self.model:
            raise RuntimeError("Whisper model not available")
        
        language = kwargs.get('language', self.language)
        beam_size = kwargs.get('beam_size', self.beam_size)
        initial_prompt = kwargs.get('initial_prompt', '')
        temperature = kwargs.get('temperature', 0.0)
        
        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
            
        )
        
        return "".join(seg.text for seg in segments).strip()
    
    def transcribe_audio_sync(self, audio_data: np.ndarray, **kwargs) -> str:
        """Synchronous wrapper for transcribe_audio."""
        return asyncio.run(self.transcribe_audio(audio_data, **kwargs))

    def listen_and_transcribe(self, sentence_callback, stop_event: threading.Event, user_has_stopped_speaking_event: threading.Event, gpu_request_queue=None, gpu_request_event=None, worker_id="STT", device_index: int = None):
        """
        Listens to the microphone, performs VAD, and calls a callback with each transcribed
        sentence as it's detected in real-time.
        """
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        audio_buffer = deque()
        is_speaking = False

        # Helper function for transcription to manage the semaphore
        def run_transcription(audio_data, **kwargs):
            gpu_request_queue.put({"type": "acquire", "priority": 1, "worker_id": worker_id})
            gpu_request_event.wait()
            try:
                # logger.info("STT: GPU acquired for transcription.")
                result = self.transcribe_audio_sync(audio_data, **kwargs)
                return result
            finally:
                # logger.info("STT: Releasing GPU.")
                gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        
        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            nonlocal is_speaking
            if stop_event.is_set(): raise sd.CallbackStop
            try:
                is_speech = vad.is_speech((indata * 32767).astype(np.int16).tobytes(), self.sample_rate)
                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        user_has_stopped_speaking_event.clear()
                        print("Speech detected.")
                    audio_buffer.extend(indata.flatten().tolist())
                else:
                    is_speaking = False
            except Exception as e:
                print(f"Error in audio callback: {e}")

        print("STT is listening for sentences...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
                                blocksize=self.frame_size, callback=audio_callback, device=device_index):
                
                last_transcript = ""
                was_speaking = False
                final_audio_buffer_len = 0
                
                while not stop_event.is_set():
                    time.sleep(0.1) # Polling rate
                    
                    if is_speaking:
                        was_speaking = True
                        # Continuously transcribe while speaking
                        if len(audio_buffer) > self.sample_rate * 1.5: # 1.5-second buffer
                            audio_np = np.array(list(audio_buffer), dtype=np.float32)
                            current_transcript = run_transcription(audio_np, beam_size=1, temperature=0.5)

                            if len(current_transcript) > len(last_transcript):
                                new_text = current_transcript[len(last_transcript):]
                                # Simple sentence detection
                                if '.' in new_text or '?' in new_text or '!' in new_text:
                                    last_transcript = current_transcript
                                    sentence_callback(current_transcript)
                                    final_audio_buffer_len = len(audio_buffer)
                    
                    elif was_speaking: # User just stopped speaking
                        was_speaking = False
                        print("End of speech detected.")
                        user_has_stopped_speaking_event.set()
                        
                        #  Only transcribe if sentence would be different -- equal audio lengths == same sentence
                        if len(audio_buffer) > 0 and len(audio_buffer) > final_audio_buffer_len: 
                            final_audio_np = np.array(list(audio_buffer), dtype=np.float32)
                            final_transcript = run_transcription(final_audio_np, beam_size=5, temperature=0.0)
                            sentence_callback(final_transcript)

                        audio_buffer.clear()
                        last_transcript = ""

        except Exception as e:
            print(f"Error in STT listen_and_transcribe: {e}")
    
    def list_available_input_devices(self):
        """List available audio input devices."""
        return list_available_input_devices()

def list_available_input_devices():
    """Lists available audio input devices using sounddevice."""
    print("Available audio input devices:")
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  Device ID {i}: {device['name']} (Sample Rate: {device['default_samplerate']})")
            input_devices.append({'id': i, 'name': device['name'], 'sample_rate': device['default_samplerate']})
    if not input_devices:
        print("No input devices found. Ensure microphone is connected and drivers are installed.")
    return input_devices
