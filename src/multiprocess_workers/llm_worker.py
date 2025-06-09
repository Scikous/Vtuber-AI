#!/usr/bin/env python3
"""
LLM (Large Language Model) Worker Process for Vtuber-AI
Handles language model processing in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
import asyncio
from multiprocessing import Queue, Event
from collections import deque

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger
from utils.file_operations import write_messages_csv


async def load_llm_model(config: dict, project_root: str, logger):
    """
    Load and initialize the LLM model asynchronously.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory path
        logger: Logger instance
        
    Returns:
        tuple: (character_model, character_name, user_name, instructions)
    """
    try:
        # Import LLM functionality
        from LLM_Wizard.model_utils import load_character
        
        # Load character info
        character_info_json_path = config.get("character_info_json", "LLM_Wizard/characters/character.json")
        if not os.path.isabs(character_info_json_path):
            character_info_json_path = os.path.join(project_root, character_info_json_path)
        
        if not os.path.exists(character_info_json_path):
            logger.error(f"Character info JSON not found at: {character_info_json_path}")
            return None, None, None, None
        
        instructions, user_name, character_name = load_character(character_info_json_path)
        
        # Get LLM configuration
        llm_class_name = config.get("llm_class_name", "VtuberExllamav2")
        llm_model_path = config.get("llm_model_path", "turboderp/Qwen2.5-VL-7B-Instruct-exl2")
        tokenizer_model_path = config.get("tokenizer_model_path", "Qwen/Qwen2.5-VL-7B-Instruct")
        llm_model_revision = config.get("llm_model_revision", "6.0bpw")
        
        # Dynamically import the LLM class
        try:
            module = __import__("LLM_Wizard.models", fromlist=[llm_class_name])
            LLMClass = getattr(module, llm_class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load LLM class '{llm_class_name}': {e}")
            return None, None, None, None
        
        # Check if the class has async load_model method
        if hasattr(LLMClass, 'load_model'):
            # Try to load the model (could be async or sync)
            load_method = LLMClass.load_model
            if asyncio.iscoroutinefunction(load_method):
                # Async load
                character_model = await load_method(
                    main_model=llm_model_path,
                    tokenizer_model=tokenizer_model_path,
                    revision=llm_model_revision,
                    character_name=character_name,
                    instructions=instructions
                )
            else:
                # Sync load
                character_model = load_method(
                    main_model=llm_model_path,
                    tokenizer_model=tokenizer_model_path,
                    revision=llm_model_revision,
                    character_name=character_name,
                    instructions=instructions
                )
        else:
            logger.error(f"LLM class '{llm_class_name}' does not have a load_model method.")
            return None, None, None, None
        
        logger.info(f"Character '{character_name}' and LLM '{llm_class_name}' loaded successfully.")
        return character_model, character_name, user_name, instructions
        
    except Exception as e:
        logger.error(f"Error loading LLM model: {e}", exc_info=True)
        return None, None, None, None


async def process_llm_message(
    character_model,
    character_name: str,
    message: str,
    context: str,
    naive_short_term_memory: deque,
    max_tokens: int,
    llm_output_queue: Queue,
    terminate_current_dialogue_event: Event,
    logger
):
    """
    Process a single message through the LLM asynchronously.
    
    Args:
        character_model: The loaded LLM model
        character_name: Name of the character
        message: Input message to process
        context: Additional context for the message
        naive_short_term_memory: Short-term memory deque
        max_tokens: Maximum tokens for generation
        llm_output_queue: Queue for TTS output
        terminate_current_dialogue_event: Event to stop current dialogue
        logger: Logger instance
        
    Returns:
        str: Generated response text
    """
    from LLM_Wizard.model_utils import contains_sentence_terminator, prompt_wrapper
    
    # Prepare prompt
    prompt = message
    if context:
        prompt = prompt_wrapper(prompt, context=context)
    
    logger.debug(f"Processing prompt: {prompt[:100]}...")
    
    # Generate response using the LLM
    try:
        # Check if dialogue_generator is async
        dialogue_gen_method = character_model.dialogue_generator
        
        if asyncio.iscoroutinefunction(dialogue_gen_method):
            # Async generator
            async_job = await dialogue_gen_method(
                prompt,
                conversation_history=naive_short_term_memory,
                max_tokens=max_tokens
            )
        else:
            # Sync generator (fallback)
            async_job = dialogue_gen_method(
                prompt,
                conversation_history=naive_short_term_memory,
                max_tokens=max_tokens
            )
        
        full_string = ""
        tts_buffer = ""
        
        # Process streaming response
        if hasattr(async_job, '__aiter__'):
            # Async iterator
            async for result in async_job:
                if terminate_current_dialogue_event.is_set():
                    logger.info("Terminate current dialogue event set. Stopping generation.")
                    if hasattr(character_model, 'cancel_dialogue_generation'):
                        if asyncio.iscoroutinefunction(character_model.cancel_dialogue_generation):
                            await character_model.cancel_dialogue_generation()
                        else:
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
                            llm_output_queue.put_nowait(text_to_send_to_tts)
                            logger.debug(f"Sent to TTS: {text_to_send_to_tts[:30]}...")
                            tts_buffer = ""
        else:
            # Regular iterator (fallback for sync generators)
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
                            llm_output_queue.put_nowait(text_to_send_to_tts)
                            logger.debug(f"Sent to TTS: {text_to_send_to_tts[:30]}...")
                            tts_buffer = ""
        
        # Send any remaining text
        if tts_buffer.strip() and not terminate_current_dialogue_event.is_set():
            remaining_text = tts_buffer.strip()
            try:
                llm_output_queue.put_nowait(remaining_text)
                logger.debug(f"Sent remaining to TTS: {remaining_text[:30]}...")
            except:
                logger.warning(f"Could not send remaining text to TTS: {remaining_text[:30]}...")
        
        return full_string.strip()
        
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}", exc_info=True)
        return ""


async def async_llm_worker(
    speech_queue: Queue,
    live_chat_queue: Queue,
    llm_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    Async LLM worker function.
    """
    logger = app_logger.get_logger("LLM-Worker")
    logger.info("Async LLM worker starting...")
    
    try:
        from LLM_Wizard.model_utils import extract_name_message
        
        config = shared_config.get("config", {})
        project_root = shared_config.get("project_root")
        conversation_log_file = shared_config.get("conversation_log_file")
        max_tokens = shared_config.get("max_tokens", 512)
        
        # Load model asynchronously
        logger.info("Loading character and LLM model...")
        character_model, character_name, user_name, instructions = await load_llm_model(
            config, project_root, logger
        )
        
        if character_model is None:
            logger.error("Failed to load LLM model. Exiting worker.")
            return
        
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
                        # Clean up potential message -> <name>: <message>
                        message_data = live_chat_queue.get_nowait()
                        if isinstance(message_data, tuple):
                            message, context = message_data
                        else:
                            message = message_data
                        message = extract_name_message(message)
                        logger.info(f"LLM received LiveChat message: {message}")
                    except:
                        # No messages available
                        await asyncio.sleep(0.01)  # Small async sleep
                        continue
                
                if not message:
                    continue
                
                # Process the message asynchronously
                full_string = await process_llm_message(
                    character_model=character_model,
                    character_name=character_name,
                    message=message,
                    context=context,
                    naive_short_term_memory=naive_short_term_memory,
                    max_tokens=max_tokens,
                    llm_output_queue=llm_output_queue,
                    terminate_current_dialogue_event=terminate_current_dialogue_event,
                    logger=logger
                )
                
                # Update memory and log conversation
                if full_string:
                    # Add to memory
                    naive_short_term_memory.append(message)
                    naive_short_term_memory.append(f"{character_name}: {full_string}")
                    
                    # Log conversation
                    if write_to_log_fn and conversation_log_file:
                        try:
                            write_to_log_fn(conversation_log_file, (user_name, message))
                            write_to_log_fn(
                                conversation_log_file,
                                (character_name, full_string))
                        except Exception as e:
                            logger.error(f"Error writing to conversation log: {e}")
                    
                    logger.info(f"Generated response: {full_string[:100]}...")
                
            except Exception as e:
                logger.error(f"Error in LLM main loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    except ImportError as e:
        logger.error(f"Failed to import LLM_Wizard: {e}")
        logger.error("LLM worker cannot start without LLM_Wizard module")
    except Exception as e:
        logger.error(f"Unhandled exception in async LLM worker: {e}", exc_info=True)
    finally:
        # Cleanup
        if 'character_model' in locals() and hasattr(character_model, 'cleanup'):
            logger.info("Cleaning up LLM model...")
            if asyncio.iscoroutinefunction(character_model.cleanup):
                await character_model.cleanup()
            else:
                character_model.cleanup()
        
        logger.info("Async LLM worker process shutting down...")


def llm_process_worker(
    speech_queue: Queue,
    live_chat_queue: Queue,
    llm_output_queue: Queue,
    terminate_event: Event,
    terminate_current_dialogue_event: Event,
    shared_config: dict
):
    """
    LLM worker process function wrapper that runs the async worker.
    
    Args:
        speech_queue: Input queue from STT
        live_chat_queue: Input queue from LiveChat
        llm_output_queue: Output queue to TTS
        terminate_event: Event to signal process termination
        terminate_current_dialogue_event: Event to stop current dialogue
        shared_config: Configuration dictionary
    """
    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async worker
        loop.run_until_complete(
            async_llm_worker(
                speech_queue,
                live_chat_queue,
                llm_output_queue,
                terminate_event,
                terminate_current_dialogue_event,
                shared_config
            )
        )
    finally:
        loop.close()