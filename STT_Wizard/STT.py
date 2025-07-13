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

    def listen_and_buffer(self, full_utterance_callback, fast_transcribe_callback, stop_event: threading.Event, user_has_stopped_speaking_event: threading.Event, device_index: int = None):
        """
        Listens to the microphone, performs VAD, and calls callbacks for both
        a fast, initial transcription and the full utterance.
        """
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        audio_buffer = deque()
        is_speaking = False
        fast_transcribe_triggered = False
        last_speech_time = time.monotonic()
        
        # Buffer for the initial chunk for fast transcription
        fast_transcribe_chunk_s = 0.5 # 500ms
        fast_transcribe_frames = int(fast_transcribe_chunk_s * self.sample_rate)
        
        def audio_callback(indata: np.ndarray, frames: int, time_info, status):
            nonlocal is_speaking, last_speech_time, fast_transcribe_triggered
            if stop_event.is_set():
                raise sd.CallbackStop

            try:
                audio_segment_int16 = (indata * 32767).astype(np.int16)
                is_speech = vad.is_speech(audio_segment_int16.tobytes(), self.sample_rate)
                
                if is_speech:
                    if not is_speaking:
                        print("Speech detected.")
                        is_speaking = True
                        fast_transcribe_triggered = False
                        user_has_stopped_speaking_event.clear()

                    audio_buffer.extend(indata.flatten().tolist())
                    last_speech_time = time.monotonic()

                    # Trigger fast transcribe after collecting a small chunk
                    if not fast_transcribe_triggered and len(audio_buffer) > fast_transcribe_frames:
                        fast_chunk = np.array(list(audio_buffer), dtype=np.float32)
                        fast_transcribe_callback(fast_chunk)
                        fast_transcribe_triggered = True

                elif is_speaking:
                    audio_buffer.extend(indata.flatten().tolist())
            except Exception as e:
                print(f"Error in audio callback: {e}")

        print("STT is listening...")
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
                                blocksize=self.frame_size, callback=audio_callback, device=device_index):
                while not stop_event.is_set():
                    time.sleep(0.1)
                    if is_speaking and (time.monotonic() - last_speech_time) > self.silence_duration_s_for_finalize:
                        print("End of speech detected.")
                        user_has_stopped_speaking_event.set()
                        
                        audio_data_np = np.array(list(audio_buffer), dtype=np.float32)
                        audio_buffer.clear()
                        is_speaking = False
                        
                        full_utterance_callback(audio_data_np)
        except Exception as e:
            print(f"Error in STT listen_and_buffer: {e}\n{traceback.format_exc()}")
        finally:
            print("STT listening stopped.")

    def listen_and_transcribe(self, callback, stop_event, loop, device_index, **kwargs):
        """Listen for audio input and transcribe it.
        
        Args:
            callback: Async callback function for transcription results
            **kwargs: Additional parameters including stop_event, loop, device_index
        """
        
        self._recognize_speech_stream(callback, stop_event, loop, device_index)
    


    def _recognize_speech_stream(self, callback, stop_event: threading.Event, 
                                    loop: asyncio.AbstractEventLoop, device_index: int = None):
        """Internal method for speech recognition streaming with fast-first approach."""
        if not self.model:
            print("Whisper model not available.")
            return

        vad = webrtcvad.Vad(self.vad_aggressiveness)

        # State variables
        vad_is_currently_speech = False
        turn_is_active = False
        last_speech_time = time.monotonic()
        turn_start_time = None
        first_word_sent = False  # Track if we've sent the fast first word
        
        current_chunk_audio = []
        committed_transcript = ""

        def safe_callback(text, is_final, is_first_word=False):
            asyncio.run_coroutine_threadsafe(callback(text, is_final, is_first_word), loop)

        def transcribe_first_word():
            """Quick transcription with fast settings to get first word ASAP."""
            nonlocal first_word_sent
            if not current_chunk_audio or first_word_sent:
                return

            # Use minimal audio for first word detection
            audio_np = np.concatenate(current_chunk_audio, axis=0).flatten()
            
            # Only need a very short segment for first word
            min_samples = int(0.2 * self.sample_rate)  # 0.5 seconds minimum
            if len(audio_np) < min_samples:
                return

            print(f"  -> Fast transcribing first word from {len(audio_np)/self.sample_rate:.2f}s audio...")
            
            # Fast settings for first word
            segments, _ = self.model.transcribe(
                audio_np,
                language=self.language,
                beam_size=1,  # Minimal beam size for speed
                temperature=0.8,  # Higher temp for faster processing
                vad_filter=False,  # Skip VAD for speed
                # No initial prompt for speed
            )

            if segments:
                first_text = "".join(seg.text for seg in segments).strip()
                if first_text:
                    print(f"  -> Fast First Word(s): '{first_text}'")
                    safe_callback(first_text, is_final=False, is_first_word=True)
                    first_word_sent = True
                    self.stt_is_listening_event.clear()


        def process_and_transcribe_chunk():
            """Process and transcribe with full accuracy settings."""
            nonlocal committed_transcript, current_chunk_audio
            if not current_chunk_audio:
                return

            # Create a NumPy array from the buffered audio chunks
            audio_np = np.concatenate(current_chunk_audio, axis=0).flatten()
            current_chunk_audio.clear()
            
            # Guard against transcribing fragments that are too short
            if len(audio_np) / self.sample_rate < self.min_chunk_s_for_interim:
                return

            print(f"  -> Full transcribing {len(audio_np)/self.sample_rate:.2f}s audio...")
            
            # Full accuracy transcription
            segments, _ = self.model.transcribe(
                audio_np,
                language=self.language,
                beam_size=self.beam_size,  # Full beam size for accuracy
                initial_prompt=committed_transcript,
                temperature=0.0,  # Low temp for accuracy
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
            print(f"  -> Full Transcript: '{committed_transcript}'")
            safe_callback(committed_transcript, is_final=False)

        def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
            """Audio callback function for sounddevice."""
            nonlocal vad_is_currently_speech, last_speech_time, turn_is_active, turn_start_time
            if stop_event.is_set(): 
                raise sd.CallbackStop

            try:
                audio_segment_int16 = (indata * 32767).astype(np.int16)
                vad_is_currently_speech = vad.is_speech(audio_segment_int16.tobytes(), self.sample_rate)

                if vad_is_currently_speech:
                    if not turn_is_active:
                        turn_is_active = True
                        first_word_sent = False  # Reset for new turn

                        print("\nSpeech started...")
                    last_speech_time = time.monotonic()
                    current_chunk_audio.append(indata.copy())
                    if not turn_start_time:
                        turn_start_time = time.monotonic()
            except Exception as e:
                print(f"Error in audio callback: {e}")

        print("STT stream started [Fast-First Model].")

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype=self.audio_dtype,
                                blocksize=self.frame_size, callback=audio_callback, device=device_index):
                
                while not stop_event.is_set():
                    current_time = time.monotonic()
                    time_since_last_speech = current_time - last_speech_time
                    
                    # Fast first word trigger - very early in the speech
                    if (turn_is_active and not first_word_sent and 
                        len(current_chunk_audio) > 0 and vad_is_currently_speech and
                        turn_start_time > 0.3):  # Just 300ms after speech starts
                        transcribe_first_word()
                        print("transcribing first word", turn_start_time)
                        turn_start_time = 0.0
                        # first_word_sent = True
                    
                    # Proactive pause trigger for full transcription
                    has_paused = not vad_is_currently_speech and turn_is_active
                    if has_paused and time_since_last_speech > self.proactive_pause_s:
                        process_and_transcribe_chunk()

                    # Finalization trigger
                    if turn_is_active and first_word_sent and time_since_last_speech > self.silence_duration_s_for_finalize:
                        # Process any final, lingering audio
                        print("finalizing")
                        self.stt_is_listening_event.set()
                        self.stt_can_finish_event.wait()
                        process_and_transcribe_chunk()
                        
                        if committed_transcript:
                            print(f"Finalizing with: '{committed_transcript}'")
                            safe_callback(committed_transcript, is_final=True)

                        
                        # Reset for the next turn
                        turn_is_active = False
                        committed_transcript = ""
                        first_word_sent = False
                        self.stt_can_finish_event.clear()

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



# if __name__ == "__main__":
#     # This script is designed to be part of a package. To run it standalone,
#     # you may need to adjust the relative imports at the top of the file.
#     # For example, change:
#     # from .utils.config import load_config
#     # to a mock version if you don't have the package structure:
#     # def load_config(): return {}
#     # And create a dummy STTBase class:
#     # class STTBase:
#     #     def __init__(self, *args, **kwargs): pass
#     #     def _load_model(self): raise NotImplementedError

#     async def main_callback(text: str, is_final: bool, is_first_word: bool):
#         """
#         Asynchronous callback function to handle transcription results.
#         This function is executed in the main thread's asyncio event loop.
#         """
#         print(f"Transcript -> is_first_word: {is_first_word}, is_final: {is_final}, text: \"{text}\"")

#     # 1. Initialize threading events required by the WhisperSTT class
#     # This event is set by STT when it detects a pause and is waiting for a signal to finalize.
#     stt_is_listening_event = threading.Event()
#     # This event is set by an external process (e.g., an LLM) to tell STT it can finalize.
#     stt_can_finish_event = threading.Event()
    
#     # 2. Instantiate the STT model
#     try:
#         whisper_model = WhisperSTT(
#             stt_is_listening_event=stt_is_listening_event,
#             stt_can_finish_event=stt_can_finish_event
#         )
#     except NameError:
#         print("Could not initialize WhisperSTT. Make sure all dependencies are installed and imports are correct.")
#         exit(1)

#     # 3. Select an audio device
#     print("\nListing available audio input devices...")
#     devices = list_available_input_devices()
#     device_index = None
#     if devices:
#         try:
#             choice = input(f"Enter the device ID to use (or press Enter for default): ")
#             if choice.strip():
#                 device_index = int(choice)
#                 print(f"Using device ID {device_index}.")
#             else:
#                 print("Using default audio device.")
#         except (ValueError, IndexError):
#             print("Invalid input. Using default audio device.")
#             device_index = None
    
#     # 4. Set up for threading and graceful shutdown
#     stop_event = threading.Event()
#     loop = asyncio.get_event_loop()

#     # 5. The listen_and_transcribe function is blocking, so it must run in a separate thread.
#     listener_thread = threading.Thread(
#         target=whisper_model.listen_and_transcribe,
#         args=(main_callback, stop_event, loop, device_index),
#         daemon=True
#     )

#     # 6. This thread simulates an external component (like an LLM) that controls when
#     # the STT should finalize its transcription.
#     def llm_interaction_simulator():
#         while not stop_event.is_set():
#             # Wait until the STT signals that it has detected a pause in speech.
#             stt_is_listening_event.wait(timeout=1.0) 
#             if stop_event.is_set():
#                 break
            
#             if stt_is_listening_event.is_set():
#                 print("\n[SIMULATOR] STT is paused. Simulating LLM response time...")
#                 time.sleep(1.5)  # Simulate work
                
#                 print("[SIMULATOR] LLM ready. Allowing STT to finalize.")
#                 stt_can_finish_event.set()  # Signal STT to proceed with finalization
#                 stt_is_listening_event.clear() # Reset the event for the next turn

#     llm_sim_thread = threading.Thread(target=llm_interaction_simulator, daemon=True)

#     print("\nStarting listener. Speak into your microphone. Press Ctrl+C to stop.")
#     try:
#         listener_thread.start()
#         llm_sim_thread.start()
        
#         # The main thread runs the asyncio event loop to execute the callbacks
#         # scheduled by the listener_thread.
#         loop.run_forever()

#     except KeyboardInterrupt:
#         print("\nCaught interrupt signal. Shutting down gracefully.")
#     finally:
#         # 7. Graceful shutdown sequence
#         stop_event.set()
        
#         # Unblock any waiting events to allow threads to exit cleanly
#         stt_is_listening_event.set()
#         stt_can_finish_event.set()
        
#         if listener_thread.is_alive():
#             listener_thread.join()
#         if llm_sim_thread.is_alive():
#             llm_sim_thread.join()

#         if loop.is_running():
#             loop.call_soon_threadsafe(loop.stop)
            
#         # Closing the loop can sometimes cause issues on exit. It's often
#         # sufficient to just stop it and let the program terminate.
#         # loop.close()
        
#         print("Shutdown complete.")