import multiprocessing as mp
import sys
import os
import time
import uuid
from collections import deque

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.common import config as app_config

def dialogue_client_worker(shutdown_event: mp.Event, 
                           stt_to_llm_queue: mp.Queue,
                           llm_request_queue: mp.Queue, 
                           llm_to_tts_queue: mp.Queue,
                           tts_request_queue: mp.Queue):
    """
    Client worker for dialogue management.
    - Manages conversation history.
    - Receives transcribed text from the STT client.
    - Constructs prompts and sends them to the GPU worker for LLM inference.
    - Receives LLM responses and sends them to the GPU worker for TTS synthesis.
    """
    logger = app_logger.get_logger("DialogueClientWorker")
    config = app_config.load_config()
    llm_settings = config.get("llm_settings", {})
    
    # Use a deque for efficient history management
    short_term_memory = deque(maxlen=llm_settings.get("short_term_memory_maxlen", 10))
    max_tokens = llm_settings.get("max_tokens", 512)

    logger.info("Dialogue client worker started.")
    
    while not shutdown_event.is_set():
        try:
            # Block and wait for the next transcription result
            stt_result = stt_to_llm_queue.get(block=True, timeout=1)
            
            if stt_result:
                user_input = stt_result.get('text')
                if not user_input: continue

                logger.info(f"Received transcription ({stt_result.get('type')}): '{user_input}'")
                
                # History is the state of conversation *before* this new input.
                current_history = list(short_term_memory)
                
                # The new prompt is just the user's latest message.
                prompt = user_input

                # Update memory for the *next* turn.
                short_term_memory.append(f"User: {user_input}")
                
                llm_request_id = str(uuid.uuid4())
                logger.info(f"Sending prompt to LLM... {prompt}")
                llm_request_queue.put({
                    "id": llm_request_id,
                    "task": "generate",
                    "prompt": prompt,
                    "history": current_history,
                    "max_tokens": max_tokens
                })

                # Consume the streaming response from the LLM
                logger.info(f"Waiting for LLM response stream... {current_history}")
                full_ai_response = ""
                while True:
                    llm_result = llm_to_tts_queue.get() # Blocking call
                    
                    if llm_result and llm_result.get('id') == llm_request_id:
                        ai_sentence = llm_result['text'].strip()
                        is_final_sentence = llm_result.get('is_final', False)

                        if ai_sentence:
                            logger.info(f"Received LLM sentence: '{ai_sentence[:50]}...'")
                            full_ai_response += (" " + ai_sentence) if full_ai_response else ai_sentence

                            # Send this sentence to TTS for synthesis
                            tts_request_id = str(uuid.uuid4())
                            tts_request_queue.put({
                                "id": tts_request_id,
                                "task": "synthesize",
                                "text": ai_sentence
                            })
                        
                        if is_final_sentence:
                            break # Exit the loop
                
                # Add the complete AI response to memory for the next turn
                if full_ai_response:
                    short_term_memory.append(f"AI: {full_ai_response}")

        except mp.queues.Empty:
            # This is expected when no new transcriptions are available
            continue
        except Exception as e:
            logger.error(f"An error occurred in the dialogue client worker: {e}", exc_info=True)
            time.sleep(1)

    logger.info("Dialogue client worker has shut down.") 