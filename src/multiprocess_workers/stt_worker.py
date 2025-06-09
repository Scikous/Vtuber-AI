# #!/usr/bin/env python3
# """
# STT (Speech-to-Text) Worker Process for Vtuber-AI
# Handles speech recognition in a dedicated process for optimal performance.
# """
# import os
# import sys
# import logging
# import time
# from multiprocessing import Queue, Event

# # Add project root to path for imports
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from utils import logger as app_logger

# def stt_process_worker(
#     speech_queue: Queue,
#     terminate_event: Event,
#     terminate_current_dialogue_event: Event,
#     is_audio_streaming_event: Event,
#     shared_config: dict
# ):
#     """
#     STT worker process function.
    
#     Args:
#         speech_queue: Output queue for recognized speech
#         terminate_event: Event to signal process termination
#         terminate_current_dialogue_event: Event to stop current dialogue
#         is_audio_streaming_event: Event indicating audio is being streamed
#         shared_config: Configuration dictionary
#     """
#     # Setup logging for this process
#     logger = app_logger.get_logger("STT-Worker")
#     logger.info("STT worker process starting...")
    
#     try:
#         # Import STT functionality
#         from STT_Wizard.STT import speech_to_text
#         from STT_Wizard.utils.stt_utils import create_stt_callback
        
#         speaker_name = shared_config.get("speaker_name", "User")
        
#         # Create modular STT callback
#         stt_callback = create_stt_callback(speech_queue, speaker_name, logger)
        

#         logger.info("STT worker process ready, starting speech recognition...")
        
#         # Main STT loop
#         while not terminate_event.is_set():
#             try:
#                 # Run speech_to_text with callback
#                 # Note: This function should be modified to be non-blocking or have timeout
#                 # speech_to_text(stt_callback, terminate_wrapper, streaming_wrapper)
#                 speech_to_text(stt_callback, terminate_current_dialogue_event, is_audio_streaming_event, device_index=None)
#                 # Small sleep to prevent excessive CPU usage
#                 time.sleep(0.01)
                
#             except Exception as e:
#                 logger.error(f"Error in STT main loop: {e}")
#                 time.sleep(0.1)  # Wait before retrying
                
#                 # Check if we should terminate
#                 if terminate_event.is_set():
#                     break
    
#     except ImportError as e:
#         logger.error(f"Failed to import STT_Wizard: {e}")
#         logger.error("STT worker cannot start without STT_Wizard module")
#     except Exception as e:
#         logger.error(f"Unhandled exception in STT worker: {e}", exc_info=True)
#     finally:
#         logger.info("STT worker process shutting down...")

#!/usr/bin/env python3
"""
STT (Speech-to-Text) Worker Process for Vtuber-AI
Handles speech recognition in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
import asyncio
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
        from STT_Wizard.utils.stt_utils import create_stt_callback
        
        speaker_name = shared_config.get("speaker_name", "User")
        
        # Create modular STT callback
        stt_callback = create_stt_callback(speech_queue, speaker_name, logger)
        
        logger.info("STT worker process ready, starting speech recognition...")
        
        # Run the async main loop
        asyncio.run(stt_main_loop(
            stt_callback,
            terminate_event,
            terminate_current_dialogue_event,
            is_audio_streaming_event,
            speech_to_text,
            logger
        ))
        
    except ImportError as e:
        logger.error(f"Failed to import STT_Wizard: {e}")
        logger.error("STT worker cannot start without STT_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in STT worker: {e}", exc_info=True)
    finally:
        logger.info("STT worker process shutting down...")

async def stt_main_loop(
    stt_callback,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    is_audio_streaming_event: Event,
    speech_to_text_func,
    logger
):
    """
    Async main loop for STT processing.
    
    Args:
        stt_callback: Callback function for STT results
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        is_audio_streaming_event: Event indicating audio is being streamed
        speech_to_text_func: The async speech_to_text function
        logger: Logger instance
    """
    while not terminate_event.is_set():
        try:
            # Create a task for speech_to_text so we can check termination events
            stt_task = asyncio.create_task(
                speech_to_text_func(
                    stt_callback, 
                    terminate_current_dialogue_event, 
                    is_audio_streaming_event, 
                    device_index=None
                )
            )
            
            # Wait for either the STT task to complete or termination
            while not stt_task.done() and not terminate_event.is_set():
                await asyncio.sleep(0.01)  # Small async sleep
                
            # Cancel the task if we're terminating
            if terminate_event.is_set() and not stt_task.done():
                stt_task.cancel()
                try:
                    await stt_task
                except asyncio.CancelledError:
                    logger.info("STT task cancelled due to termination")
                    break
            
            # If task completed naturally, get the result
            if stt_task.done():
                try:
                    await stt_task  # This will raise any exceptions from the task
                except Exception as e:
                    logger.error(f"Error in speech_to_text task: {e}")
                    
        except Exception as e:
            logger.error(f"Error in STT main loop: {e}")
            await asyncio.sleep(0.1)  # Wait before retrying
            
            # Check if we should terminate
            if terminate_event.is_set():
                break