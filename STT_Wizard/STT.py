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
MODEL_SIZE = config.get("MODEL_SIZE", "large-v3")  # Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
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

LIVE_TRANSCRIPTION_INTERVAL_S = config.get("LIVE_TRANSCRIPTION_INTERVAL_S", 0.5)
SILENCE_DURATION_S_FOR_FINALIZE = config.get("SILENCE_DURATION_S_FOR_FINALIZE", 1.0)
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

class TranscriptionResult:
    """A simple dataclass to distinguish between live and final results."""
    def __init__(self, text: str, is_final: bool):
        self.text = text
        self.is_final = is_final
    def __repr__(self):
        return f"TranscriptionResult(text='{self.text}', is_final={self.is_final})"

def recognize_speech_stream(
    callback,
    stop_event: threading.Event,
    loop: asyncio.AbstractEventLoop,
    device_index: int = None
):
    if not whisper_model:
        return

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    
    # State variables
    is_speaking = False
    last_speech_time = 0
    
    # Single audio buffer for live transcription
    live_audio_window = deque(maxlen=int(AUDIO_WINDOW_S * SAMPLE_RATE / FRAME_SIZE))
    
    def safe_queue_put(item):
        """Helper to safely put items into the asyncio queue from a thread."""
        asyncio.run_coroutine_threadsafe(callback(item), loop)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        nonlocal is_speaking, last_speech_time
        if stop_event.is_set():
            raise sd.CallbackStop
        if status:
            print(f"Warning: PortAudio status: {status}")

        try:
            audio_segment_int16 = (indata * 32767).astype(np.int16)
            vad_is_speech = vad.is_speech(audio_segment_int16.tobytes(), SAMPLE_RATE)
        except Exception:
            vad_is_speech = False
        
        if vad_is_speech:
            if not is_speaking:
                is_speaking = True
            last_speech_time = time.monotonic()
            live_audio_window.append(indata.copy())
        else:
            if is_speaking:
                is_speaking = False

    print("STT stream started [Live Mode].")
    last_live_transcript_time = 0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                             blocksize=FRAME_SIZE, callback=audio_callback, device=device_index):
            while not stop_event.is_set():
                current_time = time.monotonic()
                final = time.perf_counter()
                
                # --- Live Transcription Logic ---
                if (current_time - last_live_transcript_time > LIVE_TRANSCRIPTION_INTERVAL_S):
                    if not live_audio_window:
                        time.sleep(0.05)
                        continue
                    
                    # Transcribe the recent audio window for quick feedback
                    audio_data_np = np.concatenate(list(live_audio_window), axis=0).flatten()
                    
                    segments, _ = whisper_model.transcribe(
                        audio_data_np,
                        language=LANGUAGE,
                        beam_size=BEAM_SIZE,
                        condition_on_previous_text=False,  # Disable for faster processing
                        temperature=0.0,  # Deterministic output
                        compression_ratio_threshold=2.4,  # Early stopping
                        no_speech_threshold=0.6
                        )
                    live_text = "".join(seg.text for seg in segments).strip()
                    
                    # Determine if this should be marked as final based on silence duration
                    is_final = (not is_speaking and 
                               current_time - last_speech_time > SILENCE_DURATION_S_FOR_FINALIZE)
                    
                    if live_text:
                        # safe_queue_put(TranscriptionResult(live_text, is_final=is_final))
                        # Clear buffer after finalizing to start fresh for next utterance
                        if is_final:
                            safe_queue_put(live_text)
                            # callback(live_text)
                            final2 = time.perf_counter()
                            print(final2-final)
                            live_audio_window.clear()
                    
                    last_live_transcript_time = current_time

                time.sleep(0.01)

    except Exception as e:
        print(f"Error in STT stream: {e}\n{traceback.format_exc()}")
    finally:
        print("STT stream stopped.")

async def consumer_task(stt_queue: asyncio.Queue, terminate_event: asyncio.Event):
    """A simple consumer that prints the live and final transcripts."""
    while not terminate_event.is_set():
        try:
            result: TranscriptionResult = await asyncio.wait_for(stt_queue.get(), timeout=1.0)
            
            if not result.is_final:
                # Live feedback: use carriage return to update the same line
                # print(f"\rLive: {result.text}", end="", flush=True)
                pass
            else:
                # Final result: print on a new line
                print("\n" + "="*50)
                print(f"Final Transcript: {result.text}")
                print("="*50)
                # Here you would call your LLM or other processing logic
                # await process_with_llm(result.text)

        except asyncio.TimeoutError:
            continue


async def speech_to_text(
                         callback, 
                         terminate_event: asyncio.Event,
                         is_audio_streaming_event: asyncio.Event = asyncio.Event(),
                         device_index: int = None
):
    if not whisper_model:
        return

    stt_output_queue = asyncio.Queue()
    stop_stt_thread_event = threading.Event()
    main_loop = asyncio.get_running_loop()

    stt_thread = threading.Thread(
        target=recognize_speech_stream,
        args=(callback, stop_stt_thread_event, main_loop, device_index)
    )
    stt_thread.start()

    # Run the consumer task
    # consumer = asyncio.create_task(consumer_task(stt_output_queue, terminate_event))

    await terminate_event.wait()

    # Cleanup
    print("Termination signal received. Shutting down...")
    stop_stt_thread_event.set()
    await asyncio.sleep(0.2)
    # consumer.cancel()
    stt_thread.join(timeout=2)
    print("STT service shut down.")


if __name__ == "__main__":
    async def _stt_test_callback(speech_text):
        """This function is now the final destination for a complete utterance."""
        print(f"\n\n--- FINAL TRANSCRIPT RECEIVED ---\n'{speech_text}'\n---------------------------------\n")

    async def main_test_loop():
        print("Listing available input devices...")
        list_available_input_devices()
        
        # Using system default device for simplicity in this example
        # You can uncomment and adapt your device selection logic here if needed
        selected_device_index = None 
        print(f"Using system default input device.")
        
        print("\n--- Starting Real-Time STT Test ---")
        print("Speak into the microphone. The system will transcribe in real-time.")
        print("It will print the final text after you pause or end a sentence with punctuation.")
        print("The test will run for 30 seconds or until you press Ctrl+C.")
        
        # This event will be used to signal shutdown for the entire application
        terminate_event = asyncio.Event()

        try:
            # Start the STT service. It will run in the background.
            stt_service_task = asyncio.create_task(
                speech_to_text(
                    _stt_test_callback,
                    terminate_event,
                    device_index=None
                )
            )

            # Wait for a fixed duration for testing, or until the user interrupts
            await asyncio.sleep(30)

        except asyncio.CancelledError:
            print("\nMain loop cancelled.")
        except KeyboardInterrupt:
            print("\nTest stopped by user (Ctrl+C).")
        except Exception as e:
            print(f"An error occurred in the main test loop: {e}")
            traceback.print_exc()
        finally:
            print("--- Exiting STT test program ---")
            # Signal the STT service to shut down gracefully
            terminate_event.set()
            # Wait for the service to finish cleaning up
            await asyncio.sleep(1) 

    if WhisperModel and whisper_model:
        asyncio.run(main_test_loop())
    else:
        print("Whisper model not loaded. Cannot run STT test.")