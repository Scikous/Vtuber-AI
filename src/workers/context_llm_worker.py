import multiprocessing as mp
import time
import asyncio
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.common import config as app_config
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2

logger = app_logger.get_logger("ContextLLMWorker")
async def context_runner(shutdown_event: mp.Event, 
                         stt_stream_queue: mp.Queue, 
                         llm_control_queue: mp.Queue, 
                         user_has_stopped_speaking_event: mp.Event):
    """The main async runner for the Context LLM worker."""
    config = app_config.load_config()
    context_settings = config.get("context_llm_settings", {})
    
    # --- Load Context LLM ---
    logger.info("Context Worker: Loading model...")
    model_config = LLMModelConfig(
        main_model=context_settings.get("context_llm_model_path"),
        revision=context_settings.get("context_llm_model_revision"),
        tokenizer_model=context_settings.get("context_llm_tokenizer_model_path"),
        instructions=context_settings.get("context_llm_instructions"),
        is_vision_model=context_settings.get("context_llm_is_vision_model")
    )

    async with await VtuberExllamav2.load_model(config=model_config) as context_model:
        logger.info("✅ Context LLM model loaded.")
        
        initial_prompt = None
        tts_playback_approved = False

        while not shutdown_event.is_set():
            try:
                if user_has_stopped_speaking_event.is_set() and not tts_playback_approved:
                    logger.info("Context Worker: User stopped speaking. Approving TTS playback.")
                    tts_playback_approved = True
                    try:
                        # STT sends finalized the most clean transcript
                        # If initial prompt is not None, assume transcript meaning hasn't changed
                        if initial_prompt is not None:
                            transcript = stt_stream_queue.get(timeout=0.8)
                            logger.info("Context Worker: User stopped speaking. Sending finalized transcript")
                            llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": False, "continue_final_message": True})
                    except:
                        pass
                    initial_prompt = None

                if not stt_stream_queue.empty():
                    tts_playback_approved = False
                    transcript = stt_stream_queue.get_nowait()
                    print("CONTEXT WORKER: Received sentence:", transcript)
                    if initial_prompt is None:
                        initial_prompt = transcript
                        logger.info(f"Context Worker: Sending initial prompt: '{transcript}'")
                        llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": True, "continue_final_message": False})
                    else:
                        
                        # --- Context Check ---
                        prompt = f"""
                        Original Phrase: "{initial_prompt}"
                        New Phrase: "{transcript}"
                        """

                        change_detected = False
                        async for result in await context_model.dialogue_generator(prompt, max_tokens=5):
                            response_text = result.get("text", "").lower().strip()
                            if "yes" in response_text.lower():
                                change_detected = True
                                break
                        
                        if change_detected:
                            logger.warning(f"Context Worker: Meaning changed! New prompt: '{transcript}'")
                            llm_control_queue.put({"action": "interrupt"})
                            llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": True, "continue_final_message": False})
                            initial_prompt = None

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in context runner loop: {e}", exc_info=True)
                await asyncio.sleep(1)

def context_llm_worker(shutdown_event: mp.Event, 
                         stt_stream_queue: mp.Queue, 
                         llm_control_queue: mp.Queue, 
                         user_has_stopped_speaking_event: mp.Event):
    """Entry point for the Context LLM worker process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(context_runner(
            shutdown_event, stt_stream_queue, llm_control_queue, user_has_stopped_speaking_event
        ))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Context LLM worker has shut down.") 