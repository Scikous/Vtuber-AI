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

from .utils.stt_utils import calculate_audio_energy_rms, calculate_dbfs, count_words

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


# Energy threshold for pausing TTS (in dBFS). Adjust as needed.
# Lower values (more negative) mean more sensitive to sound.
ENERGY_THRESHOLD_DBFS = -31.0  # Example: -40dBFS is a reasonable starting point
# RMS equivalent can also be used if preferred, but dBFS is often more intuitive.
# MAX_RMS_FOR_FLOAT32 = 1.0 (for dBFS calculation with float32 audio)

# Word count for terminating current character dialogue.
USER_SPEECH_WORD_COUNT_TERMINATION = 3 # Example: if user says 3 or more words.

# Thread pool executor (remains from original code)
executor = ThreadPoolExecutor(max_workers=1)


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


def recognize_speech_sync( 
                          terminate_current_dialogue_event: asyncio.Event,
                          is_audio_streaming_event: asyncio.Event,
                          device_index: int = None): # Added device_index
    """
    Listens for a single utterance using VAD, records it, and transcribes with faster-whisper.
    This function is blocking and designed to be run in a separate thread.
    Returns transcribed text (str) or None if no speech detected or error.
    Sets terminate_current_dialogue_event based on word count of transcribed text.
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
    is_currently_speech_by_vad = False # VAD state
    speech_started_time = None  # Timestamp when speech first detected
    last_speech_time = None     # Timestamp of the last detected speech frame

    # Event to signal completion from the audio callback
    capture_done_event = threading.Event()
    # To store any exception from the callback
    callback_exception = None 
    
    # This helps manage clearing the event correctly.
    # We only clear it if we set it and VAD confirms silence or speech ends.
    pause_event_set_by_current_stt = False


    print(f"Listening for speech (VAD: {VAD_AGGRESSIVENESS}, Energy Threshold: {ENERGY_THRESHOLD_DBFS} dBFS, Word Term: {USER_SPEECH_WORD_COUNT_TERMINATION})...")

    def audio_callback(indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        nonlocal is_currently_speech_by_vad, speech_started_time, last_speech_time, voiced_frames
        nonlocal callback_exception, pause_event_set_by_current_stt
        
        if capture_done_event.is_set(): 
            return
        if status:
            # print(f"Audio callback status: {status}", flush=True) # Can be noisy
            # Consider logging critical statuses if they occur.
            # If status indicates a serious error, could set callback_exception and stop.
            # For now, we mostly rely on VAD and energy checks.
            if status.input_overflow or status.input_underflow:
                 print(f"Warning: PortAudio Input Overflow/Underflow: {status}")
            return

        try:
            current_rms = calculate_audio_energy_rms(indata)
            current_dbfs = calculate_dbfs(current_rms) # Max RMS for float32 is 1.0

            if current_dbfs > ENERGY_THRESHOLD_DBFS:
                
                if is_audio_streaming_event.is_set():
                    print(f"Energy {current_dbfs:.2f} dBFS > {ENERGY_THRESHOLD_DBFS} dBFS. Setting terminate_current_dialogue_event.")
                    # Call set_pause_event_threadsafe for thread safety if events are shared across threads.
                    # For asyncio.Event, if modified by non-asyncio thread, use loop.call_soon_threadsafe
                    # However, this callback is in a thread managed by sounddevice.
                    # The event itself is thread-safe for set/clear/is_set.
                    terminate_current_dialogue_event.set()
                    pause_event_set_by_current_stt = True 
            # else: # Energy below threshold
                # We want VAD to primarily control clearing the pause event if speech ends.
                # If energy drops but VAD still thinks it's speech, event remains set.
                # If VAD says no speech, and we had set the event, then we can clear it.
                # This is handled below in conjunction with VAD status.


            # VAD expects bytes (PCM16)
            # Convert float32 to int16 for VAD
            audio_segment_int16 = (indata * 32767).astype(np.int16)
            vad_is_speech_this_frame = vad.is_speech(audio_segment_int16.tobytes(), SAMPLE_RATE)

            current_time = time.monotonic()

            if vad_is_speech_this_frame:
                if not is_currently_speech_by_vad: 
                    # print("VAD: Speech detected...")
                    is_currently_speech_by_vad = True
                    speech_started_time = current_time
                    last_speech_time = current_time
                    for frame_data in list(ring_buffer):
                        voiced_frames.append(frame_data)
                    ring_buffer.clear()
                
                voiced_frames.append(indata.copy()) 
                last_speech_time = current_time
                
                # If VAD detects speech, ensure pause event remains set if energy threshold was also met.
                # This handles cases where energy might dip mid-speech but VAD still holds.
                # if pause_event_set_by_current_stt and not user_speaking_pause_event.is_set():
                #     user_speaking_pause_event.set()

            else: # VAD: Not speech this frame
                if is_currently_speech_by_vad: 
                    # print("VAD: Possible end of speech segment...")
                    is_currently_speech_by_vad = False 
                
                if not speech_started_time: # If speech never started according to VAD
                    ring_buffer.append(indata.copy())
                
                # if pause_event_set_by_current_stt and user_speaking_pause_event.is_set():
                #     # If VAD confirms no speech, and we had set the pause event due to energy,
                #     # it's safer to clear it.
                #     # print(f"VAD: No speech & energy low. Clearing user_speaking_pause_event.")
                #     pass
                #     pause_event_set_by_current_stt = False # We've acted on it.
            
            # Timeout and phrase completion checks (existing logic)
            if speech_started_time:
                if (current_time - speech_started_time) > MAX_PHRASE_DURATION_S:
                    print("Max phrase duration reached.")
                    capture_done_event.set()
                    # if is_audio_streaming_event.is_set(): user_speaking_pause_event.clear()
                    raise sd.CallbackStop
                if not is_currently_speech_by_vad and (current_time - last_speech_time) > SILENCE_AFTER_SPEECH_S:
                    print("Silence after speech detected by VAD.")
                    capture_done_event.set()
                    # if is_audio_streaming_event.is_set(): user_speaking_pause_event.clear()
                    raise sd.CallbackStop
            
        except sd.CallbackStop:
            raise
        except Exception as e:
            print(f"Error in audio_callback: {type(e).__name__} - {repr(e)}")
            print(traceback.format_exc())
            callback_exception = e 
            capture_done_event.set()
            raise sd.CallbackStop 

    try:
        # Use the specified device_index for the input stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=AUDIO_DTYPE, blocksize=FRAME_SIZE, callback=audio_callback, device=device_index):
            start_wait_time = time.monotonic()
            while not capture_done_event.is_set():
                if callback_exception: 
                    raise callback_exception
                
                current_wait_time = time.monotonic()
                if not speech_started_time and (current_wait_time - start_wait_time) > INITIAL_SPEECH_TIMEOUT_S:
                    print("Initial speech timeout: No speech detected by VAD.")
                    # Ensure pause event is cleared if STT times out before VAD speech ---
                    # if pause_event_set_by_current_stt and user_speaking_pause_event.is_set():
                    #     pass
                    #     pause_event_set_by_current_stt = False
                    # break 
                
                # Sleep briefly to avoid busy-waiting, allowing callback to run
                # The event wait itself is blocking, but we add a small timeout to it to check our conditions
                if capture_done_event.wait(timeout=0.1): 
                    break 
            
            capture_done_event.set() 

    except Exception as e:
        print(f"Error during audio capture stream: {e}")
        # if pause_event_set_by_current_stt and user_speaking_pause_event.is_set(): # Ensure cleanup on error
        #     pass
        return None 
    finally:
        capture_done_event.set()

    if callback_exception:
        print(f"Error from audio callback processing: {callback_exception}")
        # if pause_event_set_by_current_stt and user_speaking_pause_event.is_set(): # Ensure cleanup on error
        #     pass
        return None

    # If VAD never detected speech, but energy might have set the event ---
    # This happens if initial_speech_timeout occurred.
    if not voiced_frames:
        print("No speech frames recorded by VAD.")
        # if pause_event_set_by_current_stt and user_speaking_pause_event.is_set():
        #     # print("Clearing pause_event as no VAD speech was confirmed.")
        #     pass
        return None

    try:
        audio_data_np = np.concatenate(voiced_frames, axis=0)
        if audio_data_np.ndim > 1:
            audio_data_np = audio_data_np.flatten()
    except ValueError:
        print("No voiced frames to concatenate, or frames are empty.")
        if pause_event_set_by_current_stt: pass # Removed user_speaking_pause_event.is_set()
        return None
    
    if audio_data_np.size == 0:
        print("Concatenated audio data is empty.")
        if pause_event_set_by_current_stt: pass # Removed user_speaking_pause_event.is_set()
        return None

    print(f"Processing {len(voiced_frames)} VAD voiced frames, total duration: {len(audio_data_np)/SAMPLE_RATE:.2f}s")

    try:
        print("Transcribing audio with faster-whisper...")
        segments, info = whisper_model.transcribe(audio_data_np, language='en', beam_size=5) # TODO: Make language configurable
        
        transcribed_text = "".join(segment.text for segment in segments).strip()
        
        if transcribed_text:
            print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            print(f"Transcription: {transcribed_text}")

            # --- Word Count Termination ---
            num_words = count_words(transcribed_text)
            print(f"Word count: {num_words}")
            if num_words >= USER_SPEECH_WORD_COUNT_TERMINATION:
                if not terminate_current_dialogue_event.is_set() and is_audio_streaming_event.is_set():
                    print(f"Word count {num_words} >= {USER_SPEECH_WORD_COUNT_TERMINATION}. Setting terminate_current_dialogue_event.")
                    terminate_current_dialogue_event.set()
                    return

                # pause_event_set_by_current_stt = False # Reset for next cycle if any
            # --- Clear pause event after successful transcription if VAD confirmed speech ---
            # If we reached here, VAD confirmed speech. The user has spoken.
            # The pause event's job (to pause TTS while user might speak) is done for this utterance.
            # AudioStreamService will see this event clear and can resume TTS *if no other pause conditions exist*.
            if pause_event_set_by_current_stt:
                # print("Clearing user_speaking_pause_event after successful transcription of VAD speech.")
                pass
                # pause_event_set_by_current_stt = False # Reset for next cycle if any

            return transcribed_text
        else:
            print("Transcription resulted in empty text.")
            if pause_event_set_by_current_stt: pass # Removed user_speaking_pause_event.is_set()
            return None
    except Exception as e:
        print(f"Error during faster-whisper transcription: {e}")
        if pause_event_set_by_current_stt: pass # Removed user_speaking_pause_event.is_set()
        return None

# --- speech_to_text now accepts shared events ---
async def speech_to_text(callback, 
                         terminate_current_dialogue_event: asyncio.Event,
                         is_audio_streaming_event: asyncio.Event,
                         device_index: int = None): # Added device_index
    """
    Speech-To-Text (STT) -- Listens to microphone and turns recognized speech to text.
    Non-blocking implementation using asyncio and ThreadPoolExecutor.
    Calls the callback with the transcribed text for each utterance.
    Passes shared events to recognize_speech_sync.
    """
    if not whisper_model:
        print("STT service cannot start: Whisper model not loaded.")
        await asyncio.sleep(5) 
        return 

    # Ensure events are passed correctly
    if terminate_current_dialogue_event is None or is_audio_streaming_event is None:
        print("STT Error: Shared events not provided to speech_to_text function.")
        # Decide how to handle: raise error, log, or try to use dummy events (not recommended for prod)
        # For now, let's log and return to prevent further issues.
        # In a real app, this would be a setup error.
        await asyncio.sleep(1)
        return


    # `pause_event_set_by_current_stt` is managed within `recognize_speech_sync` per call.
    # No need to manage it here across calls to `recognize_speech_sync`.

    while True: # This loop is usually managed by the service calling speech_to_text.
                # The original speech_to_text returned after one utterance.
                # Let's keep that behavior. The STTService will loop.
        try:
            text = await asyncio.to_thread(recognize_speech_sync, 
                                           terminate_current_dialogue_event,
                                           is_audio_streaming_event,
                                           device_index) # Pass device_index
            
            if text:
                await callback(text)
                return # Return after one successful utterance, as per original design
            
            # If no text (silence, timeout, error), loop within recognize_speech_sync effectively handles retries
            # or the service calling this will loop.
            # If recognize_speech_sync returns None, this function will also effectively return (implicitly None).
            # The calling service (STTService) should handle the loop.
            # The `return` above means this `speech_to_text` function is called repeatedly by STTService.
            # If recognize_speech_sync returns None, this function returns None, STTService sleeps and calls again.
            # This seems fine.
            return # Return None if recognize_speech_sync returned None.
        
        except Exception as e:
            print(f"Unexpected error in speech_to_text async wrapper: {e}")
            # Clean up pause event if it was set by this STT cycle and an error occurred here
            # This is tricky because `pause_event_set_by_current_stt` is local to `recognize_speech_sync`
            # However, `recognize_speech_sync` has its own finally blocks and error handling for the event.
            await asyncio.sleep(1)
            return # Return None on error to allow STTService to retry.









###################

# --- Main block for testing ---
if __name__ == "__main__":
    async def _stt_test_callback(speech_text):
        print(f"CALLBACK RECEIVED: '{speech_text}'")

    # --- MODIFICATION: Create dummy events for testing ---
    test_is_audio_streaming_event = asyncio.Event() # Renamed from test_user_speaking_pause_event for clarity
    test_terminate_current_dialogue_event = asyncio.Event()

    async def main_test_loop():
        print("Listing available input devices...")
        available_devices = list_available_input_devices()
        
        selected_device_index = None
        if available_devices:
            try:
                default_input_device_idx = sd.default.device[0] # sd.default.device can be (input_idx, output_idx)
                print(f"Default input device index: {default_input_device_idx}")
                non_default_devices = [dev['id'] for dev in available_devices if dev['id'] != default_input_device_idx]
                if non_default_devices:
                    selected_device_index = non_default_devices[0]
                    print(f"Attempting to use non-default device for testing: ID {selected_device_index} - {sd.query_devices(selected_device_index)['name']}")
                elif available_devices: 
                    selected_device_index = available_devices[0]['id']
                    print(f"Attempting to use first available device for testing: ID {selected_device_index} - {sd.query_devices(selected_device_index)['name']}")
            except Exception as e:
                print(f"Could not determine specific devices, using system default. Error: {e}")
                if available_devices: # Fallback to first if specific selection failed
                    selected_device_index = available_devices[0]['id']
                    print(f"Fallback: Attempting to use first available device: ID {selected_device_index} - {sd.query_devices(selected_device_index)['name']}")
                else:
                    print("No input devices found by list_available_input_devices. Using system default.")
        else:
            print("No input devices found. Using system default.")

        print("Starting STT test. Speak into the microphone.")
        print("The program will listen for one utterance, transcribe it, then call speech_to_text again.")
        
        loop_count = 0
        try:
            while True:
                loop_count += 1
                print(f"\n--- STT Listen Cycle {loop_count} ---")
                if not whisper_model:
                    print("Whisper model not available. Exiting test.")
                    break
                
                # Reset events for each cycle in test for clarity, though STT should manage them
                # test_terminate_current_dialogue_event.clear() # Clearing is handled later in the loop for testing
                test_is_audio_streaming_event.set() # Simulate audio is streaming for the test

                await speech_to_text(_stt_test_callback, 
                                     test_terminate_current_dialogue_event,
                                     test_is_audio_streaming_event,
                                     device_index=1) # Pass selected device
                
                print(f"Is Audio Streaming Event Status (simulated): {test_is_audio_streaming_event.is_set()}")
                print(f"Terminate Event Status: {test_terminate_current_dialogue_event.is_set()}")
                print("-----------------------------------------------------")
                
                if test_terminate_current_dialogue_event.is_set():
                    print("Terminate event was set. Test loop will clear it and continue.")
                    test_terminate_current_dialogue_event.clear() # For testing, clear to allow next cycle

                # If pause event was set and not cleared by STT logic (e.g. due to timeout before VAD)
                # it might persist. STT internal logic should handle clearing it.
                # For testing, we might clear it here if we want each cycle fresh.
                # if test_is_audio_streaming_event.is_set(): # Example of how one might manage this event in a test
                #     print("Is Audio Streaming event is still set. Clearing for next test cycle if needed.")
                #     test_is_audio_streaming_event.clear()

                if loop_count >= 3: # Limit test cycles
                    print("Reached max test loops.")
                    break

                print("Listening for next utterance (Ctrl+C to stop)...")
                await asyncio.sleep(0.1) # Small delay before next listen cycle
        except KeyboardInterrupt:
            print("\nSTT test stopped by user.")
        except Exception as e:
            print(f"An error occurred in the main test loop: {e}")
            traceback.print_exc()
        finally:
            print("Exiting STT test program.")

    if WhisperModel and whisper_model: # Check if model loaded before running test
        try:
            asyncio.run(main_test_loop())
        except RuntimeError as e:
            if "Already running" in str(e): # Handle if an asyncio loop is already running (e.g. in Jupyter)
                print("Asyncio loop already running. Consider running main_test_loop() directly if in a suitable environment.")
            else:
                raise
    else:
        print("Whisper model not loaded. Cannot run STT test.")