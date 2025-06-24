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

from .utils.stt_utils import calculate_audio_energy_rms, calculate_dbfs, count_words

# Load configuration
config = load_config()


# --- Configuration for faster-whisper and VAD ---
MODEL_SIZE = config.get("MODEL_SIZE", "large-v3")  # Options: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo, "distil-whisper/distil-large-v3.5-ct2"
LANGUAGE = config.get("LANGUAGE", "en")  # Language code for transcription
BEAM_SIZE = config.get("BEAM_SIZE", 5)  # Beam size for beam search
# Determine device and compute type (GPU if available, else CPU)
try:
    import torch
    if torch.cuda.is_available():
        MODEL_DEVICE = "cuda"
        MODEL_COMPUTE_TYPE = "int8_float16" # or "float16", "int8_float16" for mixed precision
    print(f"PyTorch found. Using device: {MODEL_DEVICE} with compute type: {MODEL_COMPUTE_TYPE}")
    #maybe try if crashing
    # torch.cuda.set_per_process_memory_fraction(0.8, 0) 
    # print("Set per-process memory fraction to 80% for stability.")
except ImportError:
    MODEL_DEVICE = "cpu"
    MODEL_COMPUTE_TYPE = "int8" # or "float32" for CPU
    print("PyTorch not found. Defaulting to CPU for faster-whisper.")

# Initialize WhisperModel (globally, once)
if WhisperModel:
    print(f"Loading faster-whisper model: {MODEL_SIZE}...")
    try:
        whisper_model = WhisperModel(MODEL_SIZE, device=MODEL_DEVICE, compute_type=MODEL_COMPUTE_TYPE)
        print("faster-whisper model loaded successfully.")
         # --- WARMUP THE MODEL HERE ---
        warmup_stt() 
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        whisper_model = None # Ensure it's None if loading failed
else:
    whisper_model = None
    print("WhisperModel could not be imported. STT will not function.")

# Audio parameters for VAD and recording
SAMPLE_RATE = config.get("SAMPLE_RATE", 16000)  # Whisper models are trained on 16kHz audio
CHANNELS = config.get("CHANNELS", 1)
AUDIO_DTYPE = config.get("AUDIO_DTYPE", "float32")  # Data type for audio, faster-whisper expects float32
FRAME_DURATION_MS = config.get("FRAME_DURATION_MS", 30)  # VAD frame duration (10, 20, or 30 ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Samples per frame
VAD_AGGRESSIVENESS = config.get("VAD_AGGRESSIVENESS", 3)  # VAD aggressiveness (0-3, 3 is most aggressive)
# Timeouts and thresholds for speech detection
SILENCE_DURATION_S_FOR_FINALIZE = config.get("SILENCE_DURATION_S_FOR_FINALIZE", 1.0)
PROACTIVE_PAUSE_S = config.get("PROACTIVE_PAUSE_S", 0.05)
MIN_CHUNK_S_FOR_INTERIM = config.get("MIN_CHUNK_S_FOR_INTERIM", 0.1)
# Energy threshold for pausing TTS (in dBFS). Adjust as needed.
# Lower values (more negative) mean more sensitive to sound.
ENERGY_THRESHOLD_DBFS = config.get("ENERGY_THRESHOLD_DBFS", -40)  # Example: -40dBFS is a reasonable starting point
# RMS equivalent can also be used if preferred, but dBFS is often more intuitive.
# MAX_RMS_FOR_FLOAT32 = 1.0 (for dBFS calculation with float32 audio)

# How often we run transcription on the accumulated audio buffer
# *** NEW: Sliding Window Configuration ***
AUDIO_WINDOW_S = config.get("AUDIO_WINDOW_S", 8)
AUDIO_WINDOW_FRAMES = int(AUDIO_WINDOW_S * SAMPLE_RATE)


# Silence duration to consider a user's turn complete

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