#!/usr/bin/env python3
"""
STT (Speech-to-Text) Worker Process for Vtuber-AI
Handles speech recognition in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
from multiprocessing import Queue, Event

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger

def stt_process_worker(
    speech_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    is_audio_streaming_event: Event,
    shared_config: dict
):
    """
    STT worker process function.
    
    Args:
        speech_queue: Output queue for recognized speech
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        is_audio_streaming_event: Event indicating audio is being streamed
        shared_config: Configuration dictionary
    """
    # Setup logging for this process
    logger = app_logger.get_logger("STT-Worker")
    logger.info("STT worker process starting...")
    
    try:
        # Import STT functionality
        from STT_Wizard.STT import speech_to_text
        
        speaker_name = shared_config.get("speaker_name", "User")
        
        def stt_callback(speech):
            """Callback function for STT results."""
            try:
                # Filter out the common false positive
                if speech and speech.strip().lower() != "thank you.":
                    message = f"{speaker_name}: {speech.strip()}"
                    
                    # Put message in queue (non-blocking)
                    try:
                        speech_queue.put_nowait(message)
                        logger.debug(f"STT captured: {speech.strip()}")
                    except:
                        # Queue is full, try to make space by removing oldest item
                        try:
                            speech_queue.get_nowait()  # Remove oldest
                            speech_queue.put_nowait(message)  # Add new
                            logger.debug(f"STT captured (queue was full): {speech.strip()}")
                        except:
                            logger.warning(f"STT queue full, dropped message: {speech.strip()}")
            except Exception as e:
                logger.error(f"Error in STT callback: {e}")
        
        # Create a simple event wrapper for compatibility
        class EventWrapper:
            def __init__(self, mp_event):
                self.mp_event = mp_event
            
            def is_set(self):
                return self.mp_event.is_set()
            
            def set(self):
                self.mp_event.set()
            
            def clear(self):
                self.mp_event.clear()
        
        terminate_wrapper = EventWrapper(terminate_current_dialogue_event)
        streaming_wrapper = EventWrapper(is_audio_streaming_event)
        
        logger.info("STT worker process ready, starting speech recognition...")
        
        # Main STT loop
        while not terminate_event.is_set():
            try:
                # Run speech_to_text with callback
                # Note: This function should be modified to be non-blocking or have timeout
                # speech_to_text(stt_callback, terminate_wrapper, streaming_wrapper)
                speech_to_text(stt_callback, terminate_current_dialogue_event, is_audio_streaming_event, device_index=None)
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in STT main loop: {e}")
                time.sleep(0.1)  # Wait before retrying
                
                # Check if we should terminate
                if terminate_event.is_set():
                    break
    
    except ImportError as e:
        logger.error(f"Failed to import STT_Wizard: {e}")
        logger.error("STT worker cannot start without STT_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in STT worker: {e}", exc_info=True)
    finally:
        logger.info("STT worker process shutting down...")