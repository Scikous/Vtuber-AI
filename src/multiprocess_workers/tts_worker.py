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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
                        audio_output_queue.put_nowait(chunk)
                        logger.debug(f"TTS sent audio chunk of size {len(chunk)}")
                
                logger.debug(f"TTS completed processing: {text_input[:30]}...")
                return 0
                
            except Exception as e:
                logger.error(f"Error processing TTS item '{text_input[:30]}...': {e}")
                return 1
        
        # Use ThreadPoolExecutor for concurrent TTS processing
        with ThreadPoolExecutor(max_workers=tts_concurrency, thread_name_prefix="TTS") as executor:
            logger.info("TTS worker ready, starting text processing...")
            
            active_futures = set()
            
            # Main processing loop
            while not terminate_event.is_set():
                try:
                    # Clean up completed futures
                    completed_futures = [f for f in active_futures if f.done()]
                    for future in completed_futures:
                        active_futures.remove(future)
                        try:
                            result = future.result()
                            logger.debug(f"TTS task completed with {result} chunks")
                        except Exception as e:
                            logger.error(f"TTS task failed: {e}")
                    
                    # Check for termination of current dialogue
                    if terminate_current_dialogue_event.is_set():
                        # Cancel all active futures
                        for future in active_futures:
                            future.cancel()
                        active_futures.clear()
                        
                        # Clear the LLM output queue
                        while True:
                            try:
                                llm_output_queue.get_nowait()
                                logger.debug("Cleared LLM output from queue due to termination")
                            except:
                                break
                        
                        # Reset the event after clearing
                        terminate_current_dialogue_event.clear()
                        continue
                    
                    # Get new text from LLM
                    try:
                        text_input = llm_output_queue.get_nowait()
                        
                        if text_input and text_input.strip():
                            # Submit TTS task to thread pool
                            if len(active_futures) < tts_concurrency:
                                future = executor.submit(process_tts_item, text_input.strip())
                                active_futures.add(future)
                                logger.debug(f"Submitted TTS task: {text_input[:30]}...")
                            else:
                                # Wait for a slot to become available
                                logger.debug("TTS worker at max concurrency, waiting...")
                                time.sleep(0.01)
                    
                    except:
                        # No text available
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
                
                except Exception as e:
                    logger.error(f"Error in TTS main loop: {e}", exc_info=True)
                    time.sleep(0.1)
            
            # Wait for remaining tasks to complete
            logger.info("Waiting for remaining TTS tasks to complete...")
            for future in active_futures:
                try:
                    future.result(timeout=5)
                except Exception as e:
                    logger.error(f"Error waiting for TTS task: {e}")
    
    except ImportError as e:
        logger.error(f"Failed to import TTS_Wizard: {e}")
        logger.error("TTS worker cannot start without TTS_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in TTS worker: {e}", exc_info=True)
    finally:
        logger.info("TTS worker process shutting down...")

def create_optimized_tts_worker(
    llm_output_queue: Queue,
    audio_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    Optimized TTS worker with advanced batching and caching.
    """
    logger = app_logger.get_logger("TTS-Worker-Optimized")
    logger.info("Optimized TTS worker process starting...")
    
    try:
        from TTS_Wizard.tts_exp import XTTS_Service
        from TTS_Wizard.tts_utils import prepare_tts_params_xtts
        import hashlib
        
        config = shared_config.get("config", {})
        
        # Initialize TTS service with optimizations
        tts_service = XTTS_Service("TTS_Wizard/dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav")
        
        # Simple cache for repeated phrases
        audio_cache = {}
        max_cache_size = 100
        
        def get_cache_key(text):
            """Generate cache key for text."""
            return hashlib.md5(text.encode()).hexdigest()
        
        def process_with_cache(text_input):
            """Process TTS with caching."""
            cache_key = get_cache_key(text_input)
            
            # Check cache first
            if cache_key in audio_cache:
                logger.debug(f"TTS cache hit for: {text_input[:30]}...")
                return audio_cache[cache_key]
            
            # Generate new audio
            tts_params = prepare_tts_params_xtts(text_input)
            audio_chunks = []
            
            for chunk in tts_service.send_tts_request(**tts_params):
                if terminate_current_dialogue_event.is_set():
                    break
                if chunk:
                    audio_chunks.append(chunk)
            
            # Cache the result (if not too large)
            if len(audio_chunks) < 50:  # Reasonable size limit
                if len(audio_cache) >= max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(audio_cache))
                    del audio_cache[oldest_key]
                
                audio_cache[cache_key] = audio_chunks
                logger.debug(f"TTS cached result for: {text_input[:30]}...")
            
            return audio_chunks
        
        # Text batching for better efficiency
        text_batch = []
        batch_timeout = 0.1  # 100ms timeout for batching
        last_batch_time = time.time()
        
        def process_batch():
            """Process accumulated text batch."""
            if not text_batch:
                return
            
            # Combine short texts for more efficient processing
            combined_texts = []
            current_text = ""
            
            for text in text_batch:
                if len(current_text) + len(text) < 200:  # Combine short texts
                    current_text += " " + text if current_text else text
                else:
                    if current_text:
                        combined_texts.append(current_text)
                    current_text = text
            
            if current_text:
                combined_texts.append(current_text)
            
            # Process each combined text
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_with_cache, text) for text in combined_texts]
                
                for future in as_completed(futures):
                    try:
                        audio_chunks = future.result()
                        for chunk in audio_chunks:
                            try:
                                audio_output_queue.put_nowait(chunk)
                            except:
                                # Handle full queue
                                try:
                                    audio_output_queue.get_nowait()
                                    audio_output_queue.put_nowait(chunk)
                                except:
                                    pass
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
            
            text_batch.clear()
        
        logger.info("Optimized TTS worker ready")
        
        # Main processing loop
        while not terminate_event.is_set():
            try:
                # Handle dialogue termination
                if terminate_current_dialogue_event.is_set():
                    text_batch.clear()
                    # Clear queues
                    while True:
                        try:
                            llm_output_queue.get_nowait()
                        except:
                            break
                    terminate_current_dialogue_event.clear()
                    continue
                
                # Get text from LLM
                try:
                    text_input = llm_output_queue.get_nowait()
                    if text_input and text_input.strip():
                        text_batch.append(text_input.strip())
                        last_batch_time = time.time()
                except:
                    pass
                
                # Process batch if timeout reached or batch is full
                current_time = time.time()
                if (text_batch and 
                    (current_time - last_batch_time > batch_timeout or len(text_batch) >= 5)):
                    process_batch()
                    last_batch_time = current_time
                
                time.sleep(0.001)  # Very small sleep
            
            except Exception as e:
                logger.error(f"Error in optimized TTS loop: {e}")
                time.sleep(0.1)
        
        # Process any remaining batch
        process_batch()
    
    except Exception as e:
        logger.error(f"Error in optimized TTS worker: {e}", exc_info=True)
        # Fallback to standard implementation
        tts_process_worker(
            llm_output_queue, audio_output_queue, terminate_event,
            terminate_current_dialogue_event, shared_config
        )
    finally:
        logger.info("Optimized TTS worker process shutting down...")