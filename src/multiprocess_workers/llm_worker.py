#!/usr/bin/env python3
"""
LLM (Large Language Model) Worker Process for Vtuber-AI
Handles language model processing in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
from multiprocessing import Queue, Event
from collections import deque

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger
from utils.file_operations import write_messages_csv

def llm_process_worker(
    speech_queue: Queue,
    live_chat_queue: Queue,
    llm_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    LLM worker process function.
    
    Args:
        speech_queue: Input queue from STT
        live_chat_queue: Input queue from LiveChat
        llm_output_queue: Output queue to TTS
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        shared_config: Configuration dictionary
    """
    # Setup logging for this process
    logger = app_logger.get_logger("LLM-Worker")
    logger.info("LLM worker process starting...")
    
    try:
        # Import LLM functionality
        from LLM_Wizard.model_utils import load_character, contains_sentence_terminator, extract_name_message, prompt_wrapper
        
        config = shared_config.get("config", {})
        project_root = shared_config.get("project_root")
        conversation_log_file = shared_config.get("conversation_log_file")
        
        # Load character and LLM model
        logger.info("Loading character and LLM model...")
        
        character_info_json_path = config.get("character_info_json", "LLM_Wizard/characters/character.json")
        if not os.path.isabs(character_info_json_path):
            character_info_json_path = os.path.join(project_root, character_info_json_path)
        
        if not os.path.exists(character_info_json_path):
            logger.error(f"Character info JSON not found at: {character_info_json_path}")
            return
        
        instructions, user_name, character_name = load_character(character_info_json_path)
        
        # Load LLM model
        llm_class_name = config.get("llm_class_name", "VtuberExllamav2")
        llm_model_path = config.get("llm_model_path", "turboderp/Qwen2.5-VL-7B-Instruct-exl2")
        tokenizer_model_path = config.get("tokenizer_model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
        llm_model_revision = config.get("llm_model_revision", "6.0bpw")
        max_tokens = shared_config.get("max_tokens", 512)
        
        try:
            # Dynamically import the LLM class
            module = __import__("LLM_Wizard.models", fromlist=[llm_class_name])
            LLMClass = getattr(module, llm_class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load LLM class '{llm_class_name}': {e}")
            return
        
        # Instantiate the model
        if hasattr(LLMClass, 'load_model'):
            character_model = LLMClass.load_model(
                main_model=llm_model_path,
                tokenizer_model=tokenizer_model_path,
                revision=llm_model_revision,
                character_name=character_name,
                instructions=instructions
            )
        else:
            logger.error(f"LLM class '{llm_class_name}' does not have a recognized load_model method.")
            return
        
        logger.info(f"Character '{character_name}' and LLM '{llm_class_name}' loaded successfully.")
        
        # Initialize short-term memory
        short_term_memory_maxlen = config.get("short_term_memory_maxlen", 6)
        naive_short_term_memory = deque(maxlen=short_term_memory_maxlen)
        
        # Setup conversation logging
        write_to_log_fn = None
        if conversation_log_file:
            write_to_log_fn = write_messages_csv
            logger.info(f"Conversation logging enabled to: {conversation_log_file}")
        
        logger.info("LLM worker ready, starting message processing...")
        
        # Main processing loop
        while not terminate_event.is_set():
            try:
                message = None
                context = None
                
                # Check for messages from STT (higher priority)
                try:
                    message = speech_queue.get_nowait()
                    logger.info(f"LLM received STT message: {message}")
                except:
                    # No STT message, check live chat
                    try:
                        #clean up potential message -> <name>: <message>
                        message_data = live_chat_queue.get_nowait()
                        if isinstance(message_data, tuple):
                            message, context = message_data
                        else:
                            message = message_data
                        message = extract_name_message(message)
                        logger.info(f"LLM received LiveChat message: {message}")
                    except:
                        # No messages available
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
                        continue
                
                if not message:
                    continue
                
                # Process the message
                prompt = message
                if context:
                    prompt = prompt_wrapper(prompt, context=context)
                
                logger.debug(f"Processing prompt: {prompt[:100]}...")
                
                # Generate response using the LLM
                try:
                    # Create async job for dialogue generation
                    async_job = character_model.dialogue_generator(
                        prompt,
                        conversation_history=naive_short_term_memory,
                        max_tokens=max_tokens
                    )
                    
                    full_string = ""
                    tts_buffer = ""
                    
                    # Process streaming response
                    for result in async_job:
                        if terminate_current_dialogue_event.is_set():
                            logger.info("Terminate current dialogue event set. Stopping generation.")
                            if hasattr(character_model, 'cancel_dialogue_generation'):
                                character_model.cancel_dialogue_generation()
                            break
                        
                        chunk_text = result.get("text", "")
                        
                        if chunk_text and len(chunk_text) > 0:
                            full_string += chunk_text
                            tts_buffer += chunk_text
                            
                            # Send to TTS when we have a complete sentence
                            if contains_sentence_terminator(chunk_text):
                                text_to_send_to_tts = tts_buffer.strip()
                                if text_to_send_to_tts:
                                    try:
                                        llm_output_queue.put_nowait(text_to_send_to_tts)
                                        logger.debug(f"Sent to TTS: {text_to_send_to_tts[:30]}...")
                                    except:
                                        # Queue full, try to make space
                                        try:
                                            llm_output_queue.get_nowait()
                                            llm_output_queue.put_nowait(text_to_send_to_tts)
                                        except:
                                            logger.warning(f"LLM output queue full, dropped: {text_to_send_to_tts[:30]}...")
                                    
                                    tts_buffer = ""
                    
                    # Send any remaining text
                    if tts_buffer.strip() and not terminate_current_dialogue_event.is_set():
                        remaining_text = tts_buffer.strip()
                        try:
                            llm_output_queue.put_nowait(remaining_text)
                            logger.debug(f"Sent remaining to TTS: {remaining_text[:30]}...")
                        except:
                            logger.warning(f"Could not send remaining text to TTS: {remaining_text[:30]}...")
                    
                    # Update memory and log conversation
                    if full_string.strip():
                        # Add to memory
                        naive_short_term_memory.append(message)
                        naive_short_term_memory.append(f"{character_name}: {full_string.strip()}")
                        
                        # Log conversation
                        if write_to_log_fn and conversation_log_file:
                            try:
                                write_to_log_fn(
                                    conversation_log_file,
                                    [extract_name_message(message), full_string.strip()],
                                    [user_name, character_name]
                                )
                            except Exception as e:
                                logger.error(f"Error writing to conversation log: {e}")
                        
                        logger.info(f"Generated response: {full_string.strip()[:100]}...")
                
                except Exception as e:
                    logger.error(f"Error during LLM generation: {e}", exc_info=True)
                    continue
            
            except Exception as e:
                logger.error(f"Error in LLM main loop: {e}", exc_info=True)
                time.sleep(0.1)
    
    except ImportError as e:
        logger.error(f"Failed to import LLM_Wizard: {e}")
        logger.error("LLM worker cannot start without LLM_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in LLM worker: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'character_model' in locals() and hasattr(character_model, 'cleanup'):
            logger.info("Cleaning up LLM model...")
            character_model.cleanup()
        
        logger.info("LLM worker process shutting down...")

def create_optimized_llm_worker(
    speech_queue: Queue,
    live_chat_queue: Queue,
    llm_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    Optimized LLM worker with better memory management and batching.
    """
    logger = app_logger.get_logger("LLM-Worker-Optimized")
    logger.info("Optimized LLM worker process starting...")
    
    try:
        import threading
        from queue import Queue as ThreadQueue, Empty
        
        # Create internal queues for better message handling
        internal_message_queue = ThreadQueue(maxsize=10)
        
        def message_collector():
            """Collect messages from both input queues in a separate thread."""
            while not terminate_event.is_set():
                try:
                    # Priority: STT messages first
                    try:
                        message = speech_queue.get_nowait()
                        internal_message_queue.put(("stt", message, None))
                    except:
                        pass
                    
                    # Then live chat messages
                    try:
                        message_data = live_chat_queue.get_nowait()
                        if isinstance(message_data, tuple):
                            message, context = message_data
                        else:
                            message, context = message_data, None
                        internal_message_queue.put(("livechat", message, context))
                    except:
                        pass
                    
                    time.sleep(0.001)  # Very small sleep
                except Exception as e:
                    logger.error(f"Error in message collector: {e}")
        
        # Start message collector thread
        collector_thread = threading.Thread(target=message_collector, daemon=True)
        collector_thread.start()
        
        # Run the main LLM processing with the optimized message handling
        llm_process_worker(
            internal_message_queue, None, llm_output_queue,
            terminate_event, terminate_current_dialogue_event, shared_config
        )
    
    except Exception as e:
        logger.error(f"Error in optimized LLM worker: {e}", exc_info=True)
        # Fallback to standard implementation
        llm_process_worker(
            speech_queue, live_chat_queue, llm_output_queue,
            terminate_event, terminate_current_dialogue_event, shared_config
        )
    finally:
        logger.info("Optimized LLM worker process shutting down...")