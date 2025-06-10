#!/usr/bin/env python3
"""
Audio Worker Process for Vtuber-AI
Handles audio playback in a dedicated process using the PyAudioPlayback system.
"""
import os
import sys
import logging
import time
from multiprocessing import Queue, Event, Value
from queue import Empty

# Add project root to path for imports
# This path handling is fragile. Consider using an installable package structure for more robust path management.
try:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Assuming your custom playback system is in a structure like 'audio/pyaudio_playback.py'
    # Adjust the import path as necessary for your project structure.
    from TTS_Wizard.utils.pyaudio_playback import PyAudioPlayback
except ImportError as e:
    # If the import fails, the worker cannot function. Log this critical error.
    # A placeholder function will be defined to prevent crashing the main app on import.
    from utils.logger import conditional_print
    conditional_print(f"FATAL: Could not import PyAudioPlayback. Audio will not work. Error: {e}")
    PyAudioPlayback = None

from utils import logger as app_logger

def _clear_queue(q: Queue):
    """Helper function to empty a multiprocessing queue."""
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break

def audio_process_worker(
    audio_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    audio_playing: Value,
    is_audio_streaming_event: Event,
    shared_config: dict
):
    """
    Audio worker process function that uses the PyAudioPlayback class.

    Args:
        audio_output_queue: Queue for incoming audio chunks from TTS.
        terminate_event: Event to signal process termination.
        terminate_current_dialogue_event: Event to stop the current dialogue playback.
        audio_playing: Shared boolean indicating if audio is currently playing.
        is_audio_streaming_event: Event indicating audio is actively being streamed.
        shared_config: Configuration dictionary for the audio system.
    """
    # Setup logging for this process
    logger = app_logger.get_logger("Audio-Worker")
    
    if PyAudioPlayback is None:
        logger.error("PyAudioPlayback class not available. Audio worker cannot start.")
        terminate_event.wait() # Wait indefinitely to prevent process churn
        return

    playback_system = None
    try:
        # Initialize the custom playback system
        audio_backend_config = shared_config.get('config', {}).get('audio_backend_settings', {})

        playback_system = PyAudioPlayback(config=audio_backend_config, logger=logger)
        logger.info("Audio worker process starting with PyAudioPlayback system...")

        # Main loop for the worker process
        while not terminate_event.is_set():
            # 1. Check for dialogue interruption event (highest priority)
            if terminate_current_dialogue_event.is_set():
                logger.info("Dialogue interruption signal received. Clearing audio.")
                
                # Stop playback and clear hardware buffers
                if playback_system.is_active():
                    playback_system.stop_and_clear_internal_buffers()
                
                # Clear any pending audio chunks from the queue
                _clear_queue(audio_output_queue)
                
                # Update shared state to indicate silence
                with audio_playing.get_lock():
                    audio_playing.value = False
                is_audio_streaming_event.clear()
                
                # Acknowledge the interruption by clearing the event
                terminate_current_dialogue_event.clear()
                logger.info("Audio cleared and ready for next dialogue.")
                continue

            # 2. Try to get the next audio chunk from the queue
            try:
                # Block for a short time, allowing the loop to check termination events
                audio_chunk = audio_output_queue.get_nowait()#timeout=0.05) ##maybe maybe not

                if audio_chunk:
                    # If we receive audio, ensure the stream is open
                    if not playback_system.is_active():
                        playback_system.open_stream()
                        logger.info("Audio stream opened for new playback.")

                    # Update state to indicate we are playing
                    with audio_playing.get_lock():
                        audio_playing.value = True
                    is_audio_streaming_event.set()

                    # Write the chunk to the audio device
                    playback_system.write_chunk(audio_chunk)
            
            except Empty:
                # This is normal; it means no audio is currently available.
                # If the stream was active, it has now finished.
                if audio_playing.value:
                    logger.info("Audio stream finished.")
                    with audio_playing.get_lock():
                        audio_playing.value = False
                    is_audio_streaming_event.clear()
                
                # The playback_system stream can remain open, ready for the next chunk.
                continue
            
            except Exception as e:
                logger.error(f"Error during audio playback loop: {e}", exc_info=True)
                # In case of an error, reset state and close the stream
                with audio_playing.get_lock():
                    audio_playing.value = False
                is_audio_streaming_event.clear()
                if playback_system.is_active():
                    playback_system.close_stream()
                time.sleep(0.5) # Avoid fast error loops

    except Exception as e:
        logger.error(f"Unhandled exception in audio worker process: {e}", exc_info=True)
    finally:
        # 3. Cleanup: Ensure all resources are released on exit
        logger.info("Audio worker process shutting down...")
        if playback_system:
            try:
                logger.info("Cleaning up playback system.")
                playback_system.cleanup()
            except Exception as e:
                logger.error(f"Error during playback system cleanup: {e}")
        
        # Final state reset
        with audio_playing.get_lock():
            audio_playing.value = False
        is_audio_streaming_event.clear()
        
        logger.info("Audio worker process has shut down.")