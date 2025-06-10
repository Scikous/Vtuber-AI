#!/usr/bin/env python3
"""
TTS (Text-to-Speech) Worker Process for Vtuber-AI
Handles text-to-speech synthesis in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
import threading
from multiprocessing import Queue, Event
from queue import Queue as ThreadQueue, Empty

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger

def tts_process_worker(
    llm_output_queue: Queue,
    audio_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    TTS worker process function.
    
    Args:
        llm_output_queue: Input queue from LLM
        audio_output_queue: Output queue to Audio player
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        shared_config: Configuration dictionary
    """
    # Setup logging for this process
    logger = app_logger.get_logger("TTS-Worker")
    logger.info("TTS worker process starting...")
    
    try:
        # Import TTS functionality
        from TTS_Wizard import tts_client
        from TTS_Wizard.tts_exp import XTTS_Service
        from TTS_Wizard.tts_utils import prepare_tts_params_gpt_sovits, prepare_tts_params_xtts
        
        config = shared_config.get("config", {})
        
        # Initialize TTS service
        logger.info("Initializing TTS service...")
        
        # Use XTTS service (can be configured)
        tts_service = XTTS_Service("TTS_Wizard/dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav")
        prepare_tts_params = prepare_tts_params_xtts
        
        # TTS concurrency settings
        tts_concurrency = config.get("tts_concurrency", 3)  # Increased for multiprocessing
        
        logger.info(f"TTS service initialized with concurrency: {tts_concurrency}")
        
        def synthesize_streaming(tts_params: dict):
            """Synthesize speech from text using the TTS module."""
            try:
                return tts_service.send_tts_request(**tts_params)
            except Exception as e:
                logger.error(f"Error in TTS synthesis: {e}")
                return iter([])  # Return empty iterator on error
        
        def process_tts_item(text_input):
            """Process a single TTS request."""
            try:
                logger.debug(f"TTS processing: {text_input[:50]}...")
                
                # Prepare TTS parameters
                tts_params = prepare_tts_params(text_input)
                
                # Generate audio chunks
                for audio_chunk in synthesize_streaming(tts_params):
                    if terminate_current_dialogue_event.is_set():
                        logger.debug("TTS generation terminated by dialogue event")
                        break
                    
                    if audio_chunk:
                        audio_output_queue.put_nowait(audio_chunk)
                        logger.debug(f"TTS sent audio chunk of size {len(audio_chunk)}")
                
                logger.debug(f"TTS completed processing: {text_input[:30]}...")
                return True
                
            except Exception as e:
                logger.error(f"Error processing TTS item '{text_input[:30]}...': {e}")
                return False

        logger.info("TTS worker ready, starting text processing...")
        
        # Main processing loop - simplified for sequential processing
        while not terminate_event.is_set():
            try:
                # Check for termination of current dialogue
                if terminate_current_dialogue_event.is_set():
                    # Clear the LLM output queue
                    while True:
                        try:
                            llm_output_queue.get_nowait()
                            logger.debug("Cleared LLM output from queue due to termination")
                        except:
                            break
                    
                    # Reset the event after clearing
                    # terminate_current_dialogue_event.clear()
                    continue
                
                # Get new text from LLM
                try:
                    text_input = llm_output_queue.get_nowait()
                    
                    if text_input and text_input.strip():
                        # Process TTS request directly (sequential processing)
                        success = process_tts_item(text_input.strip())
                        if success:
                            logger.debug(f"Successfully processed TTS for: {text_input[:30]}...")
                        else:
                            logger.warning(f"Failed to process TTS for: {text_input[:30]}...")
                
                except:
                    # No text available
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
            
            except Exception as e:
                logger.error(f"Error in TTS main loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        logger.info("TTS worker process completed.")
    
    except ImportError as e:
        logger.error(f"Failed to import TTS_Wizard: {e}")
        logger.error("TTS worker cannot start without TTS_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in TTS worker: {e}", exc_info=True)
    finally:
        logger.info("TTS worker process shutting down...")
