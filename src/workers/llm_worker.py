# import multiprocessing as mp
# import sys
# import os
# import asyncio
# from collections import deque

# # Add project root to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from src.utils import logger as app_logger
# from src.utils.performance_utils import apply_system_optimizations
# from src.common import config as app_config

# # Model Imports
# from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
# from LLM_Wizard.model_utils import load_character, contains_sentence_terminator
# from TTS_Wizard.realtimetts import RealTimeTTS, pipertts_engine, coquitts_engine

# async def gpu_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, user_has_stopped_speaking_event):
#     """The main async runner for the GPU worker."""
#     logger = app_logger.get_logger("GPUWorker")
#     config = app_config.load_config()
    
#     # --- Model Loading ---
#     logger.info("GPU Worker: Loading models...")
#     apply_system_optimizations(logger)
#     terminate = "w1zt3r" #temporary, move to config as adjustable

#     tts_settings = config.get("tts_settings", {})
#     tts_engine_name = tts_settings.get("tts_engine", "piper")
#     if tts_engine_name == "piper":
#         tts_engine = pipertts_engine(tts_settings.get("model_file"), tts_settings.get("config_file"), tts_settings.get("piper_path"))
#     else: # coqui
#         tts_engine = coquitts_engine(use_deepspeed=True)
#     tts_model = RealTimeTTS(tts_engine, user_has_stopped_speaking_event=user_has_stopped_speaking_event)
#     logger.info("✅ TTS model loaded.")

#     llm_settings = config.get("llm_settings", {})
#     character_info_json_path = llm_settings.get("character_info_json")
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     char_path = os.path.join(project_root, character_info_json_path) if not os.path.isabs(character_info_json_path) else character_info_json_path
#     instructions, _, character_name = load_character(char_path)
#     model_config = LLMModelConfig(
#         main_model=llm_settings.get("llm_model_path"),
#         tokenizer_model=llm_settings.get("tokenizer_model_path"),
#         revision=llm_settings.get("llm_model_revision"),
#         character_name=character_name,
#         instructions=instructions
#     )

#     async with await VtuberExllamav2.load_model(config=model_config) as llm_model:
#         logger.info("✅ Main LLM model loaded.")
#         logger.info("GPU Worker: All models loaded. Waiting for commands.")

#         # --- TTS Consumer Task ---
#         async def tts_consumer():
#             while not shutdown_event.is_set():
#                 try:
#                     sentence = await asyncio.to_thread(llm_to_tts_queue.get)
#                     if sentence == terminate:
#                         await tts_model.tts_request_clear()
#                         continue
#                     if sentence:
#                         logger.info(f"GPU Worker: Sending to TTS: '{sentence}'")
#                         await tts_model.tts_request_async(sentence)
#                 except mp.queues.Empty:
#                     await asyncio.sleep(0.05)
#                 except Exception as e:
#                     logger.error(f"Error in TTS consumer: {e}", exc_info=True)

#         tts_consumer_task = asyncio.create_task(tts_consumer())

#         conversation_history = deque(maxlen=12)
#         # --- Main Processing Loop ---
#         while not shutdown_event.is_set():
#             try:
#                 if not llm_control_queue.empty():
#                     control_message = llm_control_queue.get_nowait()
#                     action = control_message.get("action")

#                     if action == "start":
#                         continue_final_message = control_message.get("continue_final_message")
#                         stt_message = control_message.get('prompt')
#                         # STT finished with NO major changes -- continue the original LLM output
#                         if continue_final_message:
#                             #the last message is the full LLM response
#                             prompt = conversation_history.pop()
#                             conversation_history[-1] = stt_message #update original prompt
#                         else:
#                             prompt = stt_message
#                         logger.info(f"GPU Worker: Starting generation for prompt: '{prompt}'")
#                         async_job = await llm_model.dialogue_generator(prompt, conversation_history=conversation_history, add_generation_prompt=control_message.get("add_generation_prompt"), continue_final_message=continue_final_message)
#                         full_sentence = ""
#                         full_response = ""
#                         async for result in async_job:
#                             token = result.get("text", "")
#                             if token:
#                                 full_sentence += token
#                             else:
#                                 continue
                            
#                             # Check for sentence completion and send to TTS
#                             if contains_sentence_terminator(full_sentence) and len(full_sentence) > 10:
#                                 logger.info(f"GPU Worker: Queueing complete sentence: '{full_sentence.strip()}'")
#                                 llm_to_tts_queue.put(full_sentence.strip())
#                                 full_response += full_sentence
#                                 full_sentence = "" # Reset for next sentence

#                             # Check for interrupt signal
#                             if not llm_control_queue.empty():
#                                 try:
#                                     interrupt_msg = llm_control_queue.get_nowait()
#                                     if interrupt_msg.get("action") == "interrupt":
#                                         logger.warning("GPU Worker: Interrupt received. Cancelling generation.")
#                                         while not llm_to_tts_queue.empty():
#                                             llm_to_tts_queue.get_nowait()

#                                         llm_to_tts_queue.put(terminate)
#                                         full_response = ""
#                                         await llm_model.cancel_dialogue_generation()
#                                         break
#                                 except Exception: # Should be queue.Empty, but being safe
#                                     pass # No interrupt message

#                         #save to short term memory
#                         if full_response:
#                             if not continue_final_message:
#                                 conversation_history.append(prompt)
#                             conversation_history.append(full_response)
                                    
                    
                
#                 await asyncio.sleep(0.01)

#             except Exception as e:
#                 logger.error(f"Error in GPU runner loop: {e}", exc_info=True)
#                 await asyncio.sleep(1)
        
#         tts_consumer_task.cancel()
#         try:
#             await tts_consumer_task
#         except asyncio.CancelledError:
#             logger.info("TTS consumer task cancelled.")

#     tts_model.cleanup()
#     logger.info("GPU worker has shut down.")


# def gpu_worker(shutdown_event: mp.Event, llm_control_queue: mp.Queue, llm_to_tts_queue: mp.Queue, user_has_stopped_speaking_event: mp.Event):
#     """Entry point for the GPU worker process."""
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         loop.run_until_complete(gpu_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, user_has_stopped_speaking_event))
#     except KeyboardInterrupt:
#         pass
#     finally:
#         loop.close() 



import multiprocessing as mp
import os
import asyncio
from collections import deque
from src.utils.app_utils import setup_project_root
from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations, check_gpu_memory
from src.common import config as app_config
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from LLM_Wizard.model_utils import load_character, contains_sentence_terminator
import torch

async def llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id="LLM"):
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
        
        conversation_history = deque(maxlen=12)
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
                            await check_gpu_memory(logger)
                            async_job = await llm_model.dialogue_generator(prompt, conversation_history=conversation_history, add_generation_prompt=control_message.get("add_generation_prompt"), continue_final_message=continue_final_message)
                            full_sentence = ""
                            full_response = ""
                            async for result in async_job:
                                token = result.get("text", "")
                                if token:
                                    full_sentence += token
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
                                            llm_to_tts_queue.put("w1zt3r")
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

def llm_worker(shutdown_event: mp.Event, llm_control_queue: mp.Queue, llm_to_tts_queue: mp.Queue, gpu_ready_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event):
    setup_project_root()
    worker_id = "LLM"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(llm_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, gpu_ready_event, gpu_request_queue, worker_event, worker_id))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
