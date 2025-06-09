#!/usr/bin/env python3
"""
Audio Worker Process for Vtuber-AI
Handles audio playback in a dedicated process for optimal performance.
"""
import multiprocessing
import threading
import time
import queue
from typing import Optional
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from TTS_Wizard.utils.pyaudio_playback import PyAudioPlayback
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio playback not available. Audio playback will be disabled.")

from .multiprocess_utils import setup_worker_logging
from multiprocessing import Queue, Event, Value
from ctypes import c_bool
from queue import Queue as ThreadQueue, Empty
from utils import logger as app_logger

def audio_process_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    is_audio_streaming: Event,
    shared_config: dict
):
    """
    Audio playback worker process using custom PyAudioPlayback class.
    
    Args:
        audio_output_queue: Queue containing audio data to play
        terminate_event: Event to signal worker termination
        is_audio_streaming: Event to track streaming status
        shared_config: Shared configuration dictionary
    """
    logger = setup_worker_logging("AudioWorker")
    
    if not PYAUDIO_AVAILABLE:
        logger.warning("PyAudio not available, using simple audio worker")
        return simple_audio_worker(audio_output_queue, terminate_event, is_audio_streaming, shared_config)
    
    try:
        # Get audio configuration from shared config
        audio_config = shared_config.get("audio_backend_settings", {})
        
        # Initialize PyAudioPlayback with configuration
        audio_playback = PyAudioPlayback(
            format=audio_config.get("format", "int16"),
            channels=audio_config.get("channels", 1),
            rate=audio_config.get("rate", 22050),
            chunk_size=audio_config.get("chunk_size", 1024)
        )
        
        # Initialize and open the audio stream
        audio_playback.initialize()
        audio_playback.open_stream()
        
        # Audio buffer for smooth playback
        audio_buffer = ThreadQueue(maxsize=50)
        
        def audio_consumer():
            """Thread function to consume audio from buffer and play it."""
            try:
                while not terminate_event.is_set():
                    try:
                        audio_data = audio_buffer.get(timeout=0.1)
                        if audio_data is None:  # Sentinel value
                            break
                        
                        # Play audio chunk using PyAudioPlayback
                        audio_playback.write_chunk(audio_data)
                        audio_buffer.task_done()
                        
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in audio consumer: {e}")
                        break
            except Exception as e:
                logger.error(f"Audio consumer thread error: {e}")
        
        # Start audio consumer thread
        consumer_thread = threading.Thread(target=audio_consumer, daemon=True)
        consumer_thread.start()
        
        logger.info("Audio worker started with PyAudioPlayback")
        
        # Main loop - producer
        while not terminate_event.is_set():
            try:
                # Get audio data from queue
                audio_data = audio_output_queue.get(timeout=0.1)
                
                if audio_data is None:  # Sentinel value for shutdown
                    break
                
                # Set streaming status
                is_audio_streaming.set()
                
                # Add to buffer (non-blocking)
                try:
                    audio_buffer.put_nowait(audio_data)
                except:
                    # Buffer full, skip this chunk
                    logger.warning("Audio buffer full, skipping chunk")
                
            except Empty:
                # No audio data available
                is_audio_streaming.clear()
                continue
            except Exception as e:
                logger.error(f"Error in audio worker main loop: {e}")
                break
        
        # Cleanup
        logger.info("Audio worker shutting down...")
        
        # Signal consumer thread to stop
        try:
            audio_buffer.put_nowait(None)  # Sentinel
        except:
            pass
        
        # Wait for consumer thread
        consumer_thread.join(timeout=2.0)
        
        # Close audio stream and cleanup
        audio_playback.close_stream()
        
        is_audio_streaming.clear()
        logger.info("Audio worker terminated")
        
    except Exception as e:
        logger.error(f"Audio worker error: {e}")
        is_audio_streaming.clear()

def simple_audio_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    is_audio_streaming: Event,
    shared_config: dict
):
    """
    Simple fallback audio worker when PyAudio is not available.
    """
    logger = setup_worker_logging("SimpleAudioWorker")
    logger.info("Simple audio worker started (PyAudio not available)")
    
    while not terminate_event.is_set():
        try:
            # Just consume audio data without playing
            audio_data = audio_output_queue.get(timeout=0.1)
            if audio_data is None:
                break
            
            # Simulate audio playback timing
            is_audio_streaming.set()
            time.sleep(0.1)  # Simulate playback time
            
        except Empty:
            is_audio_streaming.clear()
            continue
        except Exception as e:
            logger.error(f"Error in simple audio worker: {e}")
            break
    
    is_audio_streaming.clear()
    logger.info("Simple audio worker terminated")
def create_optimized_audio_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    is_audio_streaming: Event,
    shared_config: dict,
    buffer_size: int = 100,
    preload_chunks: int = 5
):
    """
    Create an optimized audio worker with advanced buffering and predictive loading.
    
    Args:
        audio_output_queue: Queue containing audio data to play
        terminate_event: Event to signal worker termination
        is_audio_streaming: Event to track streaming status
        shared_config: Shared configuration dictionary
        buffer_size: Maximum number of chunks in buffer
        preload_chunks: Number of chunks to preload before starting playback
    """
    logger = setup_worker_logging("OptimizedAudioWorker")
    
    if not PYAUDIO_AVAILABLE:
        logger.warning("PyAudio not available, using simple audio worker")
        return simple_audio_worker(audio_output_queue, terminate_event, is_audio_streaming, shared_config)
    
    try:
        # Get audio configuration from shared config
        audio_config = shared_config.get("audio_backend_settings", {})
        
        # Initialize PyAudioPlayback with optimized settings
        audio_playback = PyAudioPlayback(
            format=audio_config.get("format", "int16"),
            channels=audio_config.get("channels", 1),
            rate=audio_config.get("rate", 22050),
            chunk_size=min(audio_config.get("chunk_size", 1024), 512)  # Smaller chunk for lower latency
        )
        
        # Initialize and open the audio stream
        audio_playback.initialize()
        audio_playback.open_stream()
        
        # Advanced buffering system
        audio_buffer = ThreadQueue(maxsize=buffer_size)
        playback_stats = {
            'chunks_played': 0,
            'buffer_underruns': 0,
            'last_playback_time': time.time()
        }
        
        def optimized_audio_callback():
            """Optimized audio playback with predictive buffering."""
            try:
                # Preload buffer
                preload_count = 0
                while preload_count < preload_chunks and not terminate_event.is_set():
                    try:
                        audio_data = audio_output_queue.get(timeout=0.5)
                        if audio_data is None:
                            break
                        audio_buffer.put_nowait(audio_data)
                        preload_count += 1
                    except (Empty, queue.Full):
                        break
                
                logger.info(f"Preloaded {preload_count} audio chunks")
                
                while not terminate_event.is_set():
                    try:
                        # Get audio data from buffer
                        audio_data = audio_buffer.get(timeout=0.05)
                        if audio_data is None:
                            break
                        
                        # Play audio with timing optimization
                        start_time = time.time()
                        audio_playback.write_chunk(audio_data)
                        playback_time = time.time() - start_time
                        
                        # Update statistics
                        playback_stats['chunks_played'] += 1
                        playback_stats['last_playback_time'] = time.time()
                        
                        # Adaptive timing based on playback performance
                        if playback_time > 0.05:  # If playback is slow
                            time.sleep(0.001)  # Small delay to prevent overload
                        
                        # Set streaming status
                        is_audio_streaming.set()
                        
                    except Empty:
                        # Buffer underrun
                        playback_stats['buffer_underruns'] += 1
                        is_audio_streaming.clear()
                        time.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error in optimized audio callback: {e}")
                        break
                        
            except Exception as e:
                logger.error(f"Optimized audio callback error: {e}")
        
        def buffer_manager():
            """Advanced buffer management with predictive loading."""
            while not terminate_event.is_set():
                try:
                    # Check buffer level and load more data if needed
                    current_buffer_size = audio_buffer.qsize()
                    
                    # Predictive loading based on buffer level
                    if current_buffer_size < (buffer_size * 0.3):  # Buffer is getting low
                        try:
                            audio_chunk = audio_output_queue.get(timeout=0.1)
                            if audio_chunk is not None:
                                try:
                                    audio_buffer.put_nowait(audio_chunk)
                                except queue.Full:
                                    # Buffer full, this shouldn't happen with our logic
                                    logger.warning("Buffer full during predictive loading")
                        except Empty:
                            pass
                    else:
                        # Buffer is healthy, slower polling
                        time.sleep(0.01)
                        
                except Exception as e:
                    logger.error(f"Error in optimized buffer manager: {e}")
                    time.sleep(0.1)
        
        # Start optimized threads
        audio_thread = threading.Thread(target=optimized_audio_callback, daemon=True)
        buffer_thread = threading.Thread(target=buffer_manager, daemon=True)
        
        audio_thread.start()
        buffer_thread.start()
        
        logger.info(f"Optimized audio worker started (buffer_size={buffer_size}, preload={preload_chunks})")
        
        # Monitor performance
        last_stats_time = time.time()
        while not terminate_event.is_set():
            time.sleep(1.0)
            
            # Log performance statistics every 30 seconds
            if time.time() - last_stats_time > 30:
                logger.info(
                    f"Audio stats: {playback_stats['chunks_played']} chunks played, "
                    f"{playback_stats['buffer_underruns']} underruns, "
                    f"buffer size: {audio_buffer.qsize()}"
                )
                last_stats_time = time.time()
        
        # Cleanup
        logger.info("Optimized audio worker shutting down...")
        
        # Signal threads to stop
        try:
            audio_buffer.put_nowait(None)
        except:
            pass
        
        # Wait for threads
        audio_thread.join(timeout=2.0)
        buffer_thread.join(timeout=2.0)
        
        # Close audio stream and cleanup
        audio_playback.close_stream()
        
        is_audio_streaming.clear()
        logger.info("Optimized audio worker terminated")
        
    except Exception as e:
        logger.error(f"Optimized audio worker error: {e}")
        is_audio_streaming.clear()