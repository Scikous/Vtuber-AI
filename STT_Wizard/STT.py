import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

# Attempt to import necessary libraries for the new STT solution
try:
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import numpy as np
    import webrtcvad
    import collections
    import threading
    import io
    import wave
except ImportError as e:
    print(f"Error importing necessary libraries for STT: {e}")
    print("Please ensure 'faster-whisper', 'sounddevice', 'numpy', 'webrtcvad-wheels' (or 'webrtcvad') are installed.")
    # Fallback or raise error if critical components are missing
    # For now, we'll let it proceed and fail later if these are used without being imported.
    WhisperModel = None # Placeholder to avoid immediate crash if script is imported but not run

# --- Configuration for faster-whisper and VAD ---
MODEL_SIZE = "large-v3"  # Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
# Determine device and compute type (GPU if available, else CPU)
MODEL_DEVICE = "cpu"
MODEL_COMPUTE_TYPE = "int8" # or "float32" for CPU
try:
    import torch
    if torch.cuda.is_available():
        MODEL_DEVICE = "cuda"
        MODEL_COMPUTE_TYPE = "float16" # or "int8_float16" for mixed precision
    print(f"PyTorch found. Using device: {MODEL_DEVICE} with compute type: {MODEL_COMPUTE_TYPE}")
except ImportError:
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
SAMPLE_RATE = 16000  # Whisper models are trained on 16kHz audio
CHANNELS = 1
AUDIO_DTYPE = 'float32'  # Data type for audio, faster-whisper expects float32
FRAME_DURATION_MS = 30  # VAD frame duration (10, 20, or 30 ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Samples per frame
VAD_AGGRESSIVENESS = 3  # VAD aggressiveness (0-3, 3 is most aggressive)

# Timeouts and thresholds for speech detection
INITIAL_SPEECH_TIMEOUT_S = 5.0  # Max seconds to wait for speech to start
MAX_PHRASE_DURATION_S = 15.0   # Max duration of a single speech phrase
SILENCE_AFTER_SPEECH_S = 1.0   # Seconds of silence to consider a phrase ended

# Thread pool executor (remains from original code)
executor = ThreadPoolExecutor(max_workers=1)

def recognize_speech_sync():
    """
    Listens for a single utterance using VAD, records it, and transcribes with faster-whisper.
    This function is blocking and designed to be run in a separate thread.
    Returns transcribed text (str) or None if no speech detected or error.
    """
    if not whisper_model:
        print("Whisper model not loaded. Cannot perform STT.")
        return None

    vad = webrtcvad.Vad()
    try:
        vad.set_mode(VAD_AGGRESSIVENESS)
    except Exception as e:
        print(f"Failed to set VAD mode: {e}. Ensure VAD is initialized correctly.")
        return None

    # Buffers and state for VAD and recording
    # Ring buffer for audio before speech starts (catches leading sounds)
    # Buffer size: 0.5 seconds of audio before speech is detected
    ring_buffer_size_frames = int((0.5 * SAMPLE_RATE) / FRAME_SIZE)
    ring_buffer = collections.deque(maxlen=ring_buffer_size_frames)
    
    voiced_frames = []          # Holds frames identified as speech
    is_currently_speech = False # VAD state
    speech_started_time = None  # Timestamp when speech first detected
    last_speech_time = None     # Timestamp of the last detected speech frame

    # Event to signal completion from the audio callback
    capture_done_event = threading.Event()
    # To store any exception from the callback
    callback_exception = None 

    print(f"Listening for speech (VAD: {VAD_AGGRESSIVENESS}, Initial Timeout: {INITIAL_SPEECH_TIMEOUT_S}s)...")

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_currently_speech, speech_started_time, last_speech_time, voiced_frames, callback_exception
        if capture_done_event.is_set(): # If already done, ignore further callbacks
            return
        if status:
            print(f"Audio callback status: {status}", flush=True)
            # Potentially set capture_done_event on critical errors
            # callback_exception = RuntimeError(f"Audio callback error: {status}")
            # capture_done_event.set()
            # raise sd.CallbackStop # This might be too abrupt
            return

        try:
            # VAD expects bytes (PCM16)
            # Convert float32 to int16 for VAD
            audio_segment_int16 = (indata * 32767).astype(np.int16)
            vad_is_speech = vad.is_speech(audio_segment_int16.tobytes(), SAMPLE_RATE)

            current_time = time.monotonic()

            if vad_is_speech:
                if not is_currently_speech: # Speech starts
                    print("Speech detected...")
                    is_currently_speech = True
                    speech_started_time = current_time
                    last_speech_time = current_time
                    # Add pre-speech audio from ring buffer
                    for frame_data in list(ring_buffer):
                        voiced_frames.append(frame_data)
                    ring_buffer.clear()
                
                voiced_frames.append(indata.copy()) # Store original float32 data
                last_speech_time = current_time
            else: # Not speech
                if is_currently_speech: # Speech just ended
                    print("Possible end of speech segment...")
                    is_currently_speech = False # Mark as silence period after speech
                    # last_speech_time remains the time of the last speech frame
                
                # If not recording speech yet, keep filling ring_buffer
                if not speech_started_time:
                    ring_buffer.append(indata.copy())
            
            # Check for timeouts and phrase completion
            if speech_started_time:
                # Max phrase duration exceeded
                if (current_time - speech_started_time) > MAX_PHRASE_DURATION_S:
                    print("Max phrase duration reached.")
                    capture_done_event.set()
                    raise sd.CallbackStop
                # Silence after speech threshold met
                if not is_currently_speech and (current_time - last_speech_time) > SILENCE_AFTER_SPEECH_S:
                    print("Silence after speech detected.")
                    capture_done_event.set()
                    raise sd.CallbackStop
            else:
                # Initial timeout: no speech detected for too long
                # This check is better handled in the main thread waiting for the event with a timeout
                pass 

        except sd.CallbackStop:
            # This is a normal stop condition from VAD logic (max duration, silence).
            # The event should have been set by the VAD logic before raising.
            # Re-raise to allow sounddevice to stop the stream gracefully.
            raise
        except Exception as e:
            print(f"Error in audio_callback: {type(e).__name__} - {repr(e)}")
            print(traceback.format_exc())
            callback_exception = e # This will now only catch *other* unexpected exceptions
            capture_done_event.set()
            raise sd.CallbackStop # Stop the stream on actual error

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=AUDIO_DTYPE, blocksize=FRAME_SIZE, callback=audio_callback):
            # Wait for capture_done_event or initial timeout
            # The initial timeout (waiting for speech to start) is handled here.
            # The event will be set by the callback if speech is detected and ends, or max duration is hit.
            # If no speech starts at all, this wait will timeout.
            # The timeout for capture_done_event.wait() acts as the initial_speech_timeout
            # If speech starts, other timeouts (max_duration, silence_after_speech) are handled in callback.
            
            # We need a way to detect if speech *never* started within INITIAL_SPEECH_TIMEOUT_S
            # The current logic in callback sets event *after* speech. 
            # Let's adjust: the main thread waits with timeout. If timeout occurs and no speech_started_time, then it's initial timeout.
            
            # Wait for the event to be set by the callback, or for the initial timeout
            # The effective timeout here should be slightly longer than INITIAL_SPEECH_TIMEOUT_S 
            # to allow the callback to detect initial speech and then further conditions.
            # Or, the main loop checks speech_started_time after wait.
            
            # Let the stream run, and the main thread will check conditions periodically or wait on the event.
            # The `capture_done_event.wait()` will block until the callback sets it.
            # We need a timeout on this wait for the case where speech never starts.
            
            # Revised wait logic:
            start_wait_time = time.monotonic()
            while not capture_done_event.is_set():
                if callback_exception: # Propagate exception from callback
                    raise callback_exception
                
                current_wait_time = time.monotonic()
                # Check for initial speech timeout if speech hasn't started
                if not speech_started_time and (current_wait_time - start_wait_time) > INITIAL_SPEECH_TIMEOUT_S:
                    print("Initial speech timeout: No speech detected.")
                    # No need to set event here, the loop will exit, and voiced_frames will be empty.
                    break # Exit wait loop
                
                # If speech has started, rely on callback to set event for other conditions (max duration, silence)
                # Or, if we want a hard overall timeout for the listen attempt:
                # if (current_wait_time - start_wait_time) > (MAX_PHRASE_DURATION_S + INITIAL_SPEECH_TIMEOUT_S): 
                # print("Overall listen attempt timeout.")
                # break

                # Sleep briefly to avoid busy-waiting, allowing callback to run
                # The event wait itself is blocking, but we add a small timeout to it to check our conditions
                if capture_done_event.wait(timeout=0.1): # Returns true if event set, false on timeout
                    break # Event was set, exit loop
            
            # Ensure stream is stopped if loop exited due to timeout without event set by callback
            # The `with` statement handles stream closing, but sd.CallbackStop is cleaner if possible.
            # If we broke due to initial timeout, the callback might still be running. 
            # Setting the event here ensures the callback (if still processing a frame) will see it and stop.
            capture_done_event.set() # Signal callback to stop if it hasn't already

    except Exception as e:
        print(f"Error during audio capture stream: {e}")
        return None # Error during stream setup or management
    finally:
        # Ensure the event is set so any lingering callback invocation stops quickly.
        capture_done_event.set()

    if callback_exception:
        print(f"Error from audio callback processing: {callback_exception}")
        return None

    if not voiced_frames:
        print("No speech frames recorded.")
        return None

    # Concatenate voiced frames into a single NumPy array
    try:
        audio_data_np = np.concatenate(voiced_frames, axis=0)
        # Ensure audio_data_np is a 1D array as expected by faster-whisper
        if audio_data_np.ndim > 1:
            audio_data_np = audio_data_np.flatten()
    except ValueError:
        print("No voiced frames to concatenate, or frames are empty.")
        return None
    
    if audio_data_np.size == 0:
        print("Concatenated audio data is empty.")
        return None

    print(f"Processing {len(voiced_frames)} voiced frames, total duration: {len(audio_data_np)/SAMPLE_RATE:.2f}s")

    # Transcribe using faster-whisper
    try:
        print("Transcribing audio with faster-whisper...")
        segments, info = whisper_model.transcribe(audio_data_np, language='en', beam_size=5)
        
        transcribed_text = "".join(segment.text for segment in segments).strip()
        
        if transcribed_text:
            print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            print(f"Transcription: {transcribed_text}")
            return transcribed_text
        else:
            print("Transcription resulted in empty text.")
            return None
    except Exception as e:
        print(f"Error during faster-whisper transcription: {e}")
        return None

async def speech_to_text(callback):
    """
    Speech-To-Text (STT) -- Listens to microphone and turns recognized speech to text.
    Non-blocking implementation using asyncio and ThreadPoolExecutor.
    Calls the callback with the transcribed text for each utterance.
    """
    if not whisper_model:
        print("STT service cannot start: Whisper model not loaded.")
        # Optionally, could raise an error or enter a retry loop for model loading.
        await asyncio.sleep(5) # Wait before trying again or exiting
        return # Or raise an exception

    while True:
        try:
            # Run the synchronous speech recognition in a separate thread
            text = await asyncio.to_thread(recognize_speech_sync)
            
            if text:
                await callback(text)
                # The original design returns after one successful callback.
                # This means the outer loop (e.g., in main or service) calls speech_to_text repeatedly.
                return 
            
            # If no text (e.g., silence, timeout, error in sync function), loop continues.
            # Small delay to prevent busy-waiting if recognize_speech_sync returns None quickly.
            await asyncio.sleep(0.1)
        
        except Exception as e:
            print(f"Unexpected error in speech_to_text async loop: {e}")
            await asyncio.sleep(1)  # Wait a bit before retrying the loop

# --- Main block for testing --- (largely same as original)
if __name__ == "__main__":
    # Basic callback for testing
    async def _stt_test_callback(speech_text):
        print(f"CALLBACK RECEIVED: '{speech_text}'")
        # Example: Add to a queue or process further
        # For this test, we'll just print.
        # If you have a test queue like before:
        # await test_speech_queue.put(speech_text.strip())
        # print(list(test_speech_queue._queue))

    # If using a test queue:
    # test_speech_queue = asyncio.Queue(maxsize=4)

    print("Starting STT test. Speak into the microphone.")
    print("The program will listen for one utterance, transcribe it, then exit the speech_to_text call.")
    print("The while True loop in __main__ will then call it again.")
    
    # Loop to continuously listen for speech utterances
    try:
        while True:
            if not whisper_model:
                print("Whisper model not available. Exiting test.")
                break
            asyncio.run(speech_to_text(_stt_test_callback))
            print("-----------------------------------------------------")
            print("Listening for next utterance (Ctrl+C to stop)...")
            # Add a small delay if speech_to_text returns very quickly (e.g. on immediate timeout)
            # asyncio.run(asyncio.sleep(0.1)) # This would need to be part of an async main
    except KeyboardInterrupt:
        print("\nSTT test stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main test loop: {e}")
    finally:
        print("Exiting STT test program.")