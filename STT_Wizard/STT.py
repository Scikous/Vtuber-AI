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
    
    def __init__(self, model_size: str = None, language: str = None, device: str = None, compute_type: str = None, **stt_settings):
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
        self.frame_duration_ms = self.config.get("FRAME_DURATION_MS", 30)
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.vad_aggressiveness = self.config.get("VAD_AGGRESSIVENESS", 3)
        
        # Speech detection parameters
        self.silence_duration_s_for_finalize = self.config.get("SILENCE_DURATION_S_FOR_FINALIZE", 1.0)
        self.proactive_pause_s = self.config.get("PROACTIVE_PAUSE_S", 0.05)
        self.min_chunk_s_for_interim = self.config.get("MIN_CHUNK_S_FOR_INTERIM", 0.1)
        self.energy_threshold_dbfs = self.config.get("ENERGY_THRESHOLD_DBFS", -40)
        
        # Audio window configuration
        self.audio_window_s = self.config.get("AUDIO_WINDOW_S", 8)
        self.audio_window_frames = int(self.audio_window_s * self.sample_rate)
        
        # Initialize the parent class
        super().__init__(model_path=self.model_size, device=self.device, compute_type=self.compute_type)
    
    def _load_model(self):
        """Load the Whisper model."""
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
    
    def listen_and_transcribe(self, callback, stop_event, loop, device_index, **kwargs):
        """Listen for audio input and transcribe it.
        
        Args:
            callback: Async callback function for transcription results
            **kwargs: Additional parameters including stop_event, loop, device_index
        """
        
        self._recognize_speech_stream(callback, stop_event, loop, device_index)
    
    def _recognize_speech_stream(self, callback, stop_event: threading.Event, 
                                     loop: asyncio.AbstractEventLoop, device_index: int = None):
        """Internal method for speech recognition streaming."""
        if not self.model:
            print("Whisper model not available.")
            return

        vad = webrtcvad.Vad(self.vad_aggressiveness)

        # State variables
        vad_is_currently_speech = False
        turn_is_active = False
        last_speech_time = time.monotonic()
        
        current_chunk_audio = []
        committed_transcript = ""

        def safe_callback(text, is_final):
            asyncio.run_coroutine_threadsafe(callback(text, is_final), loop)

        def process_and_transcribe_chunk():
            """Process and transcribe the current audio buffer."""
            nonlocal committed_transcript, current_chunk_audio
            if not current_chunk_audio:
                return

            # Create a NumPy array from the buffered audio chunks
            audio_np = np.concatenate(current_chunk_audio, axis=0).flatten()
            current_chunk_audio.clear()
            
            # Guard against transcribing fragments that are too short
            if len(audio_np) / self.sample_rate < self.min_chunk_s_for_interim:
                return

            print(f"  -> Transcribing {len(audio_np)/self.sample_rate:.2f}s audio...")
            
            # Transcribe the audio
            segments, _ = self.model.transcribe(
                audio_np,
                language=self.language,
                beam_size=self.beam_size,
                initial_prompt=committed_transcript,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=250),
            )

            new_text = "".join(seg.text for seg in segments).strip()

            # Handle Whisper repeating the prompt
            if new_text and not new_text.lower().strip().startswith(committed_transcript.lower().strip()):
                committed_transcript += " " + new_text
            else:
                committed_transcript = new_text

            committed_transcript = committed_transcript.strip()
            print(f"  -> Interim Transcript: '{committed_transcript}'")
            safe_callback(committed_transcript, is_final=False)

        def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
            """Audio callback function for sounddevice."""
            nonlocal vad_is_currently_speech, last_speech_time, turn_is_active
            if stop_event.is_set(): 
                raise sd.CallbackStop

            try:
                audio_segment_int16 = (indata * 32767).astype(np.int16)
                vad_is_currently_speech = vad.is_speech(audio_segment_int16.tobytes(), self.sample_rate)

                if vad_is_currently_speech:
                    if not turn_is_active:
                        turn_is_active = True
                        print("\nSpeech started...")
                    last_speech_time = time.monotonic()
                    current_chunk_audio.append(indata.copy())
            except Exception as e:
                print(f"Error in audio callback: {e}")

        print("STT stream started [Optimized Single-Threaded Model].")

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
                                 blocksize=self.frame_size, callback=audio_callback, device=device_index):
                
                while not stop_event.is_set():
                    current_time = time.monotonic()
                    time_since_last_speech = current_time - last_speech_time
                    
                    # Proactive pause trigger
                    has_paused = not vad_is_currently_speech and turn_is_active
                    if has_paused and time_since_last_speech > self.proactive_pause_s:
                        process_and_transcribe_chunk()

                    # Finalization trigger
                    if turn_is_active and time_since_last_speech > self.silence_duration_s_for_finalize:
                        # Process any final, lingering audio
                        process_and_transcribe_chunk()
                        
                        if committed_transcript:
                            print(f"Finalizing with: '{committed_transcript}'")
                            safe_callback(committed_transcript, is_final=True)
                        
                        # Reset for the next turn
                        turn_is_active = False
                        committed_transcript = ""

                    # Sleep for polling rate
                    time.sleep(0.02)

        except Exception as e:
            print(f"Error in STT stream: {e}\n{traceback.format_exc()}")
        finally:
            print("STT stream stopped.")
    
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


def warmup_stt():
    """
    Warms up the STT model by running a dummy transcription.

    This function should be called once after the Whisper model is loaded.
    It processes a short silent audio clip, which triggers the Just-In-Time (JIT)
    compilation and kernel caching in CTranslate2 (the backend for faster-whisper).
    This significantly reduces the latency of the very first "real" transcription.
    """
    if not whisper_model:
        print("Warmup skipped: Whisper model is not available.")
        return

    print("Warming up the STT model...")
    start_time = time.monotonic()
    
    # Create a short, silent audio array. 0.5 seconds of silence is plenty.
    # The audio format must match what the model expects: 16kHz, float32.
    silent_audio = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    
    # Run a transcription on the silent audio
    # We don't need the result, just the act of running it is the warmup.
    try:
        # Use simple parameters for the warmup
        _, _ = whisper_model.transcribe(silent_audio, beam_size=1, language=LANGUAGE)
        
        end_time = time.monotonic()
        print(f"STT model warmed up in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred during STT model warmup: {e}")


def recognize_speech_stream(
    callback,
    stop_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
    device_index: int = None
):
    """
    An optimized, single-threaded speech recognition stream.

    This model prioritizes stability and simplicity. It accepts the inherent
    latency of the whisper.transcribe() call (~200ms) as a floor and builds
    the most responsive system possible around it.
    """
    if not whisper_model:
        print("Whisper model not available.")
        return

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # --- State Variables ---
    vad_is_currently_speech = False
    turn_is_active = False
    last_speech_time = time.monotonic()
    
    current_chunk_audio = []
    committed_transcript = ""

    def safe_callback(text, is_final):
        asyncio.run_coroutine_threadsafe(callback(text, is_final), loop)

    def process_and_transcribe_chunk():
        """
        Takes the current audio buffer, transcribes it, and updates the state.
        This is the primary "work" function.
        """
        nonlocal committed_transcript, current_chunk_audio
        if not current_chunk_audio:
            return

        # Create a NumPy array from the buffered audio chunks
        audio_np = np.concatenate(current_chunk_audio, axis=0).flatten()
        current_chunk_audio.clear()
        
        # Guard against transcribing fragments that are too short
        if len(audio_np) / SAMPLE_RATE < MIN_CHUNK_S_FOR_INTERIM:
            return

        print(f"  -> Transcribing {len(audio_np)/SAMPLE_RATE:.2f}s audio...")
        
        # --- The Blocking Call ---
        # We accept that this call will block the loop for ~200-400ms.
        segments, _ = whisper_model.transcribe(
            audio_np,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            initial_prompt=committed_transcript,
            temperature=0.0,
            vad_filter=True, # Use Whisper's VAD for a final cleanup
            vad_parameters=dict(min_silence_duration_ms=250),
        )

        new_text = "".join(seg.text for seg in segments).strip()

        # Handle Whisper repeating the prompt
        if new_text and not new_text.lower().strip().startswith(committed_transcript.lower().strip()):
            committed_transcript += " " + new_text
        else:
            committed_transcript = new_text

        committed_transcript = committed_transcript.strip()
        print(f"  -> Interim Transcript: '{committed_transcript}'")
        safe_callback(committed_transcript, is_final=False)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """This function is called by sounddevice. It's kept extremely light."""
        nonlocal vad_is_currently_speech, last_speech_time, turn_is_active
        if stop_event.is_set(): raise sd.CallbackStop

        try:
            audio_segment_int16 = (indata * 32767).astype(np.int16)
            vad_is_currently_speech = vad.is_speech(audio_segment_int16.tobytes(), SAMPLE_RATE)

            if vad_is_currently_speech:
                if not turn_is_active:
                    turn_is_active = True
                    print("\nSpeech started...")
                last_speech_time = time.monotonic()
                current_chunk_audio.append(indata.copy())
        except Exception as e:
            print(f"Error in audio callback: {e}")

    print("STT stream started [Optimized Single-Threaded Model].")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=AUDIO_DTYPE,
                             blocksize=FRAME_SIZE, callback=audio_callback, device=device_index):
            
            while not stop_event.is_set():
                current_time = time.monotonic()
                time_since_last_speech = current_time - last_speech_time
                
                # --- Proactive Pause Trigger ---
                # This is the core of our low-latency logic. If the user has paused
                # briefly, we immediately process the audio collected so far.
                has_paused = not vad_is_currently_speech and turn_is_active
                if has_paused and time_since_last_speech > PROACTIVE_PAUSE_S:
                    process_and_transcribe_chunk()

                # --- Finalization Trigger ---
                # If the turn has been silent for a longer period, we finalize it.
                if turn_is_active and time_since_last_speech > SILENCE_DURATION_S_FOR_FINALIZE:
                    # Process any final, lingering audio
                    process_and_transcribe_chunk()
                    
                    if committed_transcript:
                        print(f"Finalizing with: '{committed_transcript}'")
                        safe_callback(committed_transcript, is_final=True)
                    
                    # Reset for the next turn
                    turn_is_active = False
                    committed_transcript = ""

                # The sleep duration determines the polling rate for our logic.
                # A lower value makes the pause detection more responsive.
                time.sleep(0.02)

    except Exception as e:
        print(f"Error in STT stream: {e}\n{traceback.format_exc()}")
    finally:
        print("STT stream stopped.")