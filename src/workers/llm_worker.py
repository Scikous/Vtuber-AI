import multiprocessing as mp
import os
import asyncio
from collections import deque
from src.utils.app_utils import setup_project_root
from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations, async_check_gpu_memory
from src.common import config as app_config
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from LLM_Wizard.model_utils import load_character, contains_sentence_terminator

async def llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id="LLM", llm_output_display_queue=None):
    logger = app_logger.get_logger("LLMWorker")
    config = app_config.load_config()
    apply_system_optimizations(logger, use_cuda=True)
    
    logger.info("LLM Worker: Loading model...")
    llm_settings = config.get("llm_settings", {})
    character_info_json_path = llm_settings.get("character_info_json")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    char_path = os.path.join(project_root, character_info_json_path) if not os.path.isabs(character_info_json_path) else character_info_json_path
    instructions, _, character_name = load_character(char_path)
    model_config = LLMModelConfig(
        main_model=llm_settings.get("llm_model_path"),
        tokenizer_model=llm_settings.get("tokenizer_model_path"),
        revision=llm_settings.get("llm_model_revision"),
        character_name=character_name,
        instructions=instructions
    )
    max_tokens = llm_settings.get("max_tokens", 512)
    conversation_history_maxlen = llm_settings.get("conversation_history_maxlen", 12)
    TERMINATE_OUTPUT = config.get("TERMINATE_OUTPUT", "w1zt3r")

    async def yield_and_queue_sentence(sentence: str):
        """
        Temporarily releases the GPU semaphore to safely queue a sentence.
        This function is the key to preventing the deadlock.
        """
        if not sentence:
            return

        logger.info(f"LLM Worker: Releasing GPU to queue sentence: '{sentence}'")
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        try:
            # This put is now safe, as the TTS worker can acquire the released GPU.
            # We use a loop with a small sleep to make it responsive to shutdowns.
            while not shutdown_event.is_set():
                try:
                    llm_to_tts_queue.put(sentence, timeout=0.1)
                    logger.info(f"LLM Worker: Sentence queued successfully.")
                    return
                except mp.queues.Full:
                    # Queue is still full, wait a moment and let other processes run.
                    await asyncio.sleep(0.05)
        finally:
            # VERY IMPORTANT: Re-acquire the semaphore to resume generation.
            gpu_request_queue.put({"type": "acquire", "priority": 3, "worker_id":  worker_id})
            worker_event.wait()
            logger.info("LLM Worker: Re-acquiring GPU semaphore to continue generation.")
    
    async with await VtuberExllamav2.load_model(config=model_config) as llm_model:
        logger.info("✅ Main LLM model loaded.")
        gpu_ready_event.set()
        logger.info("LLM Worker: Waiting for commands.")
        
        conversation_history = deque(maxlen=conversation_history_maxlen)
        while not shutdown_event.is_set():
            try:
                if not llm_control_queue.empty():
                    control_message = llm_control_queue.get_nowait()
                    action = control_message.get("action")
                    if action == "start":
                        continue_final_message = control_message.get("continue_final_message", False)
                        stt_message = control_message.get("prompt")
                        if continue_final_message:
                            prompt = conversation_history.pop()
                            conversation_history[-1] = stt_message
                        else:
                            prompt = stt_message

                        logger.info(f"LLM Worker: Starting generation for prompt: '{prompt}'")
                        try:
                            gpu_request_queue.put({"type": "acquire", "priority": 2, "worker_id":  worker_id})
                            worker_event.wait()
                            await async_check_gpu_memory(logger)
                            async_job = await llm_model.dialogue_generator(prompt, conversation_history=conversation_history, max_tokens=max_tokens, add_generation_prompt=control_message.get("add_generation_prompt"), continue_final_message=continue_final_message)
                            full_sentence = ""
                            full_response = ""
                            async for result in async_job:
                                token = result.get("text", "")
                                if token:
                                    full_sentence += token
                                    if llm_output_display_queue:
                                        llm_output_display_queue.put(token)
                                else:
                                    continue

                                
                                if contains_sentence_terminator(full_sentence) and len(full_sentence) > 10:
                                    logger.info(f"LLM Worker: Queueing sentence: '{full_sentence.strip()}'")
                                    # llm_to_tts_queue.put(full_sentence.strip())
                                    await yield_and_queue_sentence(full_sentence.strip())
                                    full_response += full_sentence
                                    full_sentence = ""
                                
                                if not llm_control_queue.empty():
                                    try:
                                        interrupt_msg = llm_control_queue.get_nowait()
                                        if interrupt_msg.get("action") == "interrupt":
                                            logger.warning("LLM Worker: Interrupt received. Cancelling generation.")
                                            while not llm_to_tts_queue.empty():
                                                llm_to_tts_queue.get_nowait()
                                            llm_to_tts_queue.put(TERMINATE_OUTPUT)
                                            full_response = ""
                                            full_sentence = ""
                                            await llm_model.cancel_dialogue_generation()
                                            break
                                    except mp.queues.Empty:
                                        pass
                                
                            if full_response:
                                if not continue_final_message:
                                    conversation_history.append(prompt)
                                conversation_history.append(full_response)
                        finally:
                            # Ensure the semaphore is ALWAYS released, even if errors occur.
                            logger.info("LLM Worker: Generation finished. Releasing GPU semaphore.")
                            gpu_request_queue.put({"type": "release", "worker_id": worker_id})
                            await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in LLM runner loop: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    logger.info("LLM worker has shut down.")

def llm_worker(shutdown_event: mp.Event, llm_control_queue: mp.Queue, llm_to_tts_queue: mp.Queue, gpu_ready_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event, llm_output_display_queue: mp.Queue):
    setup_project_root()
    worker_id = "LLM"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id, llm_output_display_queue))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
