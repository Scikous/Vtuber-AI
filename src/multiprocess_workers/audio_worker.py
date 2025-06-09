#!/usr/bin/env python3
"""
Audio Worker Process for Vtuber-AI
Handles audio playback in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
import threading
from multiprocessing import Queue, Event, Value
from ctypes import c_bool
from queue import Queue as ThreadQueue, Empty

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger

def audio_process_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    audio_playing: Value,
    is_audio_streaming_event: Event,
    shared_config: dict
):
    """
    Audio worker process function.
    
    Args:
        audio_output_queue: Input queue from TTS
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        audio_playing: Shared boolean indicating if audio is playing
        is_audio_streaming_event: Event indicating audio streaming status
        shared_config: Configuration dictionary
    """
    # Setup logging for this process
    logger = app_logger.get_logger("Audio-Worker")
    logger.info("Audio worker process starting...")
    
    try:
        import pyaudio
        import wave
        import io
        import numpy as np
        from collections import deque
        
        # Audio configuration
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050  # Common TTS output rate
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Get default output device
        try:
            output_device_index = p.get_default_output_device_info()['index']
            logger.info(f"Using default audio output device: {output_device_index}")
        except:
            output_device_index = None
            logger.warning("Could not get default output device, using system default")
        
        # Audio buffer for smooth playback
        audio_buffer = deque()
        buffer_lock = threading.Lock()
        
        def audio_callback():
            """Audio playback callback running in separate thread."""
            stream = None
            try:
                # Open audio stream
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=CHUNK
                )
                
                logger.info("Audio stream opened successfully")
                
                while not terminate_event.is_set():
                    try:
                        # Check if we should stop current dialogue
                        if terminate_current_dialogue_event.is_set():
                            # Clear the buffer
                            with buffer_lock:
                                audio_buffer.clear()
                            
                            # Update shared state
                            with audio_playing.get_lock():
                                audio_playing.value = False
                            is_audio_streaming_event.clear()
                            
                            # Wait for the event to be cleared
                            while terminate_current_dialogue_event.is_set():
                                time.sleep(0.01)
                            continue
                        
                        # Get audio data from buffer
                        audio_data = None
                        with buffer_lock:
                            if audio_buffer:
                                audio_data = audio_buffer.popleft()
                        
                        if audio_data:
                            # Update shared state
                            with audio_playing.get_lock():
                                audio_playing.value = True
                            is_audio_streaming_event.set()
                            
                            # Play audio chunk
                            try:
                                # Ensure audio data is in correct format
                                if isinstance(audio_data, bytes):
                                    stream.write(audio_data)
                                else:
                                    # Convert numpy array or other formats to bytes
                                    if hasattr(audio_data, 'tobytes'):
                                        stream.write(audio_data.tobytes())
                                    else:
                                        # Try to convert to numpy array first
                                        audio_array = np.array(audio_data, dtype=np.int16)
                                        stream.write(audio_array.tobytes())
                                
                                logger.debug(f"Played audio chunk of size {len(audio_data)}")
                            
                            except Exception as e:
                                logger.error(f"Error playing audio chunk: {e}")
                        
                        else:
                            # No audio data available
                            with audio_playing.get_lock():
                                audio_playing.value = False
                            
                            # Only clear streaming event if buffer is truly empty
                            with buffer_lock:
                                if not audio_buffer:
                                    is_audio_streaming_event.clear()
                            
                            time.sleep(0.01)  # Small sleep when no audio
                    
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
                        time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error setting up audio stream: {e}")
            finally:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                        logger.info("Audio stream closed")
                    except:
                        pass
        
        def buffer_manager():
            """Manage audio buffer from TTS output queue."""
            while not terminate_event.is_set():
                try:
                    # Get audio from TTS queue
                    try:
                        audio_chunk = audio_output_queue.get_nowait()
                        
                        if audio_chunk:
                            # Add to buffer
                            with buffer_lock:
                                audio_buffer.append(audio_chunk)
                            
                            logger.debug(f"Added audio chunk to buffer, buffer size: {len(audio_buffer)}")
                            
                            # Limit buffer size to prevent memory issues
                            with buffer_lock:
                                while len(audio_buffer) > 50:  # Max 50 chunks in buffer
                                    audio_buffer.popleft()
                                    logger.debug("Removed old audio chunk from buffer")
                    
                    except:
                        # No audio available
                        time.sleep(0.001)  # Very small sleep
                
                except Exception as e:
                    logger.error(f"Error in buffer manager: {e}")
                    time.sleep(0.1)
        
        # Start audio threads
        audio_thread = threading.Thread(target=audio_callback, daemon=True, name="AudioPlayback")
        buffer_thread = threading.Thread(target=buffer_manager, daemon=True, name="AudioBuffer")
        
        audio_thread.start()
        buffer_thread.start()
        
        logger.info("Audio worker threads started")
        
        # Keep the main thread alive
        while not terminate_event.is_set():
            time.sleep(0.1)
        
        logger.info("Waiting for audio threads to finish...")
        audio_thread.join(timeout=2)
        buffer_thread.join(timeout=2)
    
    except ImportError as e:
        logger.error(f"Failed to import required audio modules: {e}")
        logger.error("Audio worker cannot start without PyAudio and NumPy")
        # Fallback to simple audio handling
        simple_audio_worker(
            audio_output_queue, terminate_event, terminate_current_dialogue_event,
            audio_playing, is_audio_streaming_event, shared_config
        )
    except Exception as e:
        logger.error(f"Unhandled exception in audio worker: {e}", exc_info=True)
    finally:
        # Cleanup PyAudio
        try:
            if 'p' in locals():
                p.terminate()
        except:
            pass
        
        logger.info("Audio worker process shutting down...")
        
def create_optimized_audio_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    audio_playing: Value,
    is_audio_streaming_event: Event,
    shared_config: dict
):
    """
    Optimized audio worker with advanced buffering and low-latency playback.
    """
    logger = app_logger.get_logger("Audio-Worker-Optimized")
    logger.info("Optimized audio worker starting...")
    
    try:
        import pyaudio
        import numpy as np
        from collections import deque
        import threading
        
        # Optimized audio configuration
        CHUNK = 512  # Smaller chunks for lower latency
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050
        
        # Advanced buffering
        audio_buffer = deque()
        buffer_lock = threading.RLock()  # Reentrant lock
        target_buffer_size = 5  # Target number of chunks in buffer
        
        # Initialize PyAudio with optimizations
        p = pyaudio.PyAudio()
        
        def optimized_audio_callback():
            """Optimized audio callback with predictive buffering."""
            stream = None
            try:
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=None  # Use blocking mode for better control
                )
                
                logger.info("Optimized audio stream opened")
                
                while not terminate_event.is_set():
                    try:
                        if terminate_current_dialogue_event.is_set():
                            with buffer_lock:
                                audio_buffer.clear()
                            with audio_playing.get_lock():
                                audio_playing.value = False
                            is_audio_streaming_event.clear()
                            continue
                        
                        # Smart buffer management
                        with buffer_lock:
                            buffer_size = len(audio_buffer)
                            
                            if buffer_size > 0:
                                audio_data = audio_buffer.popleft()
                                
                                # Update state
                                with audio_playing.get_lock():
                                    audio_playing.value = True
                                is_audio_streaming_event.set()
                                
                                # Play audio with error handling
                                try:
                                    if isinstance(audio_data, bytes):
                                        stream.write(audio_data)
                                    else:
                                        # Convert to proper format
                                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                        stream.write(audio_array.tobytes())
                                except Exception as e:
                                    logger.error(f"Audio playback error: {e}")
                            
                            else:
                                # No audio in buffer
                                with audio_playing.get_lock():
                                    audio_playing.value = False
                                is_audio_streaming_event.clear()
                                time.sleep(0.005)  # Very short sleep
                    
                    except Exception as e:
                        logger.error(f"Error in optimized audio callback: {e}")
                        time.sleep(0.01)
            
            finally:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except:
                        pass
        
        def optimized_buffer_manager():
            """Optimized buffer manager with predictive loading."""
            while not terminate_event.is_set():
                try:
                    # Check buffer level and load accordingly
                    with buffer_lock:
                        current_buffer_size = len(audio_buffer)
                    
                    # Load more audio if buffer is low
                    if current_buffer_size < target_buffer_size:
                        try:
                            audio_chunk = audio_output_queue.get_nowait()
                            if audio_chunk:
                                with buffer_lock:
                                    audio_buffer.append(audio_chunk)
                                    # Prevent buffer overflow
                                    while len(audio_buffer) > target_buffer_size * 3:
                                        audio_buffer.popleft()
                        except:
                            pass
                    
                    time.sleep(0.001)  # Minimal sleep for efficiency
                
                except Exception as e:
                    logger.error(f"Error in optimized buffer manager: {e}")
                    time.sleep(0.01)
        
        # Start optimized threads
        audio_thread = threading.Thread(target=optimized_audio_callback, daemon=True)
        buffer_thread = threading.Thread(target=optimized_buffer_manager, daemon=True)
        
        audio_thread.start()
        buffer_thread.start()
        
        logger.info("Optimized audio worker ready")
        
        # Main thread monitoring
        while not terminate_event.is_set():
            time.sleep(0.1)
        
        # Cleanup
        audio_thread.join(timeout=1)
        buffer_thread.join(timeout=1)
        
    except Exception as e:
        logger.error(f"Error in optimized audio worker: {e}", exc_info=True)
        # Fallback to standard implementation
        audio_process_worker(
            audio_output_queue, terminate_event, terminate_current_dialogue_event,
            audio_playing, is_audio_streaming_event, shared_config
        )
    finally:
        try:
            if 'p' in locals():
                p.terminate()
        except:
            pass
        logger.info("Optimized audio worker shutting down...")