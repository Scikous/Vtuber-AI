import asyncio
import time
from collections import deque

# Attempt to import necessary libraries for the new STT solution
try:
    from faster_whisper import WhisperModel
    from .utils.config import load_config
    from .utils.stt_utils import calculate_audio_energy_rms, calculate_dbfs, count_words, STTBase
    import sounddevice as sd
    import numpy as np
    import webrtcvad
    
except ImportError as e:
    print(f"Error importing necessary libraries for STT: {e}")
    print("Please ensure 'faster-whisper', 'sounddevice', 'numpy', 'webrtcvad-wheels' (or 'webrtcvad') are installed.")
    # Fallback or raise error if critical components are missing
    WhisperModel = None # Placeholder to avoid immediate crash if script is imported but not run

class WhisperSTT(STTBase):
    """Whisper-based Speech-to-Text implementation using faster-whisper."""
    
    def __init__(self, model_size: str = None, language: str = None, device: str = None, compute_type: str = None, **stt_settings):
        # Load configuration
        self.config = stt_settings if stt_settings else load_config()
        
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
        self.min_chunk_s_for_interim = self.config.get("MIN_CHUNK_S_FOR_INTERIM", 0.1)
        self.audio_window_s = self.config.get("AUDIO_WINDOW_S", 6)
        
        self.model = None
        self.stream = None 

    
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
            self._warmup_model()
        except Exception as e:
            print(f"Error loading faster-whisper model: {e}")
            self.model = None
    
    def _warmup_model(self):
        """Warm up the STT model with a dummy transcription."""
        if not self.model:
            print("Warmup skipped: Whisper model is not available.")
            return
        print("Warming up the STT model...")
        start_time = time.monotonic()
        silent_audio = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        try:
            _, _ = self.model.transcribe(silent_audio, beam_size=1, language=self.language)
            end_time = time.monotonic()
            print(f"STT model warmed up in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred during STT model warmup: {e}")
    
    async def transcribe_audio(self, audio_data: np.ndarray, **kwargs) -> str:
        """Transcribe audio data using Whisper."""
        if not self.model:
            raise RuntimeError("Whisper model not available")
        language = kwargs.get('language', self.language)
        beam_size = kwargs.get('beam_size', self.beam_size)
        temperature = kwargs.get('temperature', 0.0)
        segments, _ = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
        )
        return "".join(seg.text for seg in segments).strip()
    
    def transcribe_audio_sync(self, audio_data: np.ndarray, **kwargs) -> str:
        """Synchronous wrapper for transcribe_audio."""
        return asyncio.run(self.transcribe_audio(audio_data, **kwargs))
    
    def listen_and_transcribe(self, shutdown_event, sentence_callback, transcription_func=None, on_speech_start=None, on_speech_end=None, device_index: int = None, mute_event=None):
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        audio_buffer = deque()
        is_speaking = False
        transcription_func = transcription_func or self.transcribe_audio_sync

        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            nonlocal is_speaking
            try:
                if mute_event and mute_event.is_set():
                    return
                is_speech = vad.is_speech((indata * 32767).astype(np.int16).tobytes(), self.sample_rate)
                if is_speech:
                    if not is_speaking:
                        is_speaking = True
                        if on_speech_start:
                            on_speech_start()
                        print("Speech detected.")
                    audio_buffer.extend(indata.flatten().tolist())
                else:
                    if is_speaking:
                        is_speaking = False
                        if on_speech_end:
                            on_speech_end()
                        print("End of speech detected.")
            except Exception as e:
                print(f"Error in audio callback: {e}")

        print("STT is listening for sentences...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
                                blocksize=self.frame_size, callback=audio_callback, device=device_index):
                last_transcript = ""
                final_audio_buffer_len = 0
                while not shutdown_event.is_set():  # Assuming external termination via process shutdown
                    time.sleep(0.1)  # Polling rate
                    if is_speaking:
                        # Interim transcription
                        if len(audio_buffer) > self.sample_rate * 1.5:  # 1.5-second buffer
                            audio_np = np.array(list(audio_buffer), dtype=np.float32)
                            current_transcript = transcription_func(audio_np, beam_size=1, temperature=0.5)
                            if current_transcript and len(current_transcript) > len(last_transcript):
                                new_text = current_transcript[len(last_transcript):]
                                if '.' in new_text or '?' in new_text or '!' in new_text:
                                    last_transcript = current_transcript
                                    sentence_callback(current_transcript)
                                    final_audio_buffer_len = len(audio_buffer)
                    else:
                        # Final transcription when speech ends
                        if len(audio_buffer) > 0 and len(audio_buffer) > final_audio_buffer_len:
                            final_audio_np = np.array(list(audio_buffer), dtype=np.float32)
                            final_transcript = transcription_func(final_audio_np, beam_size=5, temperature=0.0)
                            if final_transcript:
                                sentence_callback(final_transcript)
                        audio_buffer.clear()
                        last_transcript = ""
                        final_audio_buffer_len = 0
        except Exception as e:
            print(f"Error in listen_and_transcribe: {e}")

    # def listen_and_transcribe(self, sentence_callback, transcription_func=None, on_speech_start=None, on_speech_end=None, device_index: int = None, shutdown_event=None, mute_event=None):
    #     vad = webrtcvad.Vad(self.vad_aggressiveness)
    #     audio_buffer = deque()
    #     is_speaking = False
    #     transcription_func = transcription_func or self.transcribe_audio_sync
    #     # ADDED: State to handle clearing buffer once when mute is activated
    #     was_muted = False

    #     def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    #         nonlocal is_speaking, was_muted
    #         if mute_event and mute_event.is_set():
    #             if not was_muted:
    #                 audio_buffer.clear() # Clear buffer on first detection of mute
    #                 was_muted = True
    #             return
    #         was_muted = False # Reset flag if not muted
    #         try:
    #             is_speech = vad.is_speech((indata * 32767).astype(np.int16).tobytes(), self.sample_rate)
    #             if is_speech:
    #                 if not is_speaking:
    #                     is_speaking = True
    #                     if on_speech_start:
    #                         on_speech_start()
    #                     print("Speech detected.")
    #                 audio_buffer.extend(indata.flatten().tolist())
    #             else:
    #                 if is_speaking:
    #                     is_speaking = False
    #                     if on_speech_end:
    #                         on_speech_end()
    #                     print("End of speech detected.")
    #         except Exception as e:
    #             print(f"Error in audio callback: {e}")

    #     print("STT is listening for sentences...")
    #     try:
    #         with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
    #                             blocksize=self.frame_size, callback=audio_callback, device=device_index):
    #             last_transcript = ""
    #             final_audio_buffer_len = 0
    #             while shutdown_event.is_set():
    #                 time.sleep(0.1)  # Polling rate
    #                 if is_speaking:
    #                     # Interim transcription
    #                     if len(audio_buffer) > self.sample_rate * 1.5:  # 1.5-second buffer
    #                         audio_np = np.array(list(audio_buffer), dtype=np.float32)
    #                         current_transcript = transcription_func(audio_np, beam_size=1, temperature=0.5)
    #                         if current_transcript and len(current_transcript) > len(last_transcript):
    #                             new_text = current_transcript[len(last_transcript):]
    #                             if '.' in new_text or '?' in new_text or '!' in new_text:
    #                                 last_transcript = current_transcript
    #                                 sentence_callback(current_transcript)
    #                                 final_audio_buffer_len = len(audio_buffer)
    #                 else:
    #                     # Final transcription when speech ends
    #                     if len(audio_buffer) > 0 and len(audio_buffer) > final_audio_buffer_len:
    #                         final_audio_np = np.array(list(audio_buffer), dtype=np.float32)
    #                         final_transcript = transcription_func(final_audio_np, beam_size=5, temperature=0.0)
    #                         if final_transcript:
    #                             sentence_callback(final_transcript)
    #                     audio_buffer.clear()
    #                     last_transcript = ""
    #                     final_audio_buffer_len = 0
    #     except Exception as e:
    #         # Don't log exceptions if it's part of a clean shutdown
    #         if not (shutdown_event and shutdown_event.is_set()):
    #             print(f"Error in listen_and_transcribe: {e}")

    def list_available_input_devices(self):
        """List available audio input devices."""
        return list_available_input_devices()

def list_available_input_devices():
    """Lists available audio input devices usingmaking devices."""
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