import multiprocessing as mp
import sys
import os
import asyncio
from collections import deque

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations
from src.common import config as app_config

# Model Imports
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from LLM_Wizard.model_utils import load_character, contains_sentence_terminator
from TTS_Wizard.realtimetts import RealTimeTTS, pipertts_engine, coquitts_engine

async def gpu_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, tts_go_event):
    """The main async runner for the GPU worker."""
    logger = app_logger.get_logger("GPUWorker")
    config = app_config.load_config()
    
    # --- Model Loading ---
    logger.info("GPU Worker: Loading models...")
    apply_system_optimizations(logger)
    terminate = "w1zt3r" #temporary, move to config as adjustable

    tts_settings = config.get("tts_settings", {})
    tts_engine_name = tts_settings.get("tts_engine", "piper")
    if tts_engine_name == "piper":
        tts_engine = pipertts_engine(tts_settings.get("model_file"), tts_settings.get("config_file"), tts_settings.get("piper_path"))
    else: # coqui
        tts_engine = coquitts_engine(use_deepspeed=True)
    tts_model = RealTimeTTS(tts_engine, tts_go_event=tts_go_event)
    logger.info("✅ TTS model loaded.")

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

    async with await VtuberExllamav2.load_model(config=model_config) as llm_model:
        logger.info("✅ Main LLM model loaded.")
        logger.info("GPU Worker: All models loaded. Waiting for commands.")

        # --- TTS Consumer Task ---
        async def tts_consumer():
            while not shutdown_event.is_set():
                try:
                    sentence = await asyncio.to_thread(llm_to_tts_queue.get)
                    if sentence == terminate:
                        await tts_model.tts_request_clear()
                        continue
                    if sentence:
                        logger.info(f"GPU Worker: Sending to TTS: '{sentence}'")
                        await tts_model.tts_request_async(sentence)
                except mp.queues.Empty:
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.error(f"Error in TTS consumer: {e}", exc_info=True)

        tts_consumer_task = asyncio.create_task(tts_consumer())

        conversation_history = deque(maxlen=12)
        # --- Main Processing Loop ---
        while not shutdown_event.is_set():
            try:
                if not llm_control_queue.empty():
                    control_message = llm_control_queue.get_nowait()
                    action = control_message.get("action")

                    if action == "start":
                        continue_final_message = control_message.get("continue_final_message")
                        stt_message = control_message.get('prompt')
                        # STT finished with NO major changes -- continue the original LLM output
                        if continue_final_message:
                            #the last message is the full LLM response
                            prompt = conversation_history.pop()
                            conversation_history[-1] = stt_message #update original prompt
                        else:
                            prompt = stt_message
                        logger.info(f"GPU Worker: Starting generation for prompt: '{prompt}'")
                        async_job = await llm_model.dialogue_generator(prompt, conversation_history=conversation_history, add_generation_prompt=control_message.get("add_generation_prompt"), continue_final_message=continue_final_message)
                        full_sentence = ""
                        full_response = ""
                        async for result in async_job:
                            token = result.get("text", "")
                            if token:
                                full_sentence += token
                            else:
                                continue
                            
                            # Check for sentence completion and send to TTS
                            if contains_sentence_terminator(full_sentence) and len(full_sentence) > 10:
                                logger.info(f"GPU Worker: Queueing complete sentence: '{full_sentence.strip()}'")
                                llm_to_tts_queue.put(full_sentence.strip())
                                full_response += full_sentence
                                full_sentence = "" # Reset for next sentence

                            # Check for interrupt signal
                            if not llm_control_queue.empty():
                                try:
                                    interrupt_msg = llm_control_queue.get_nowait()
                                    if interrupt_msg.get("action") == "interrupt":
                                        logger.warning("GPU Worker: Interrupt received. Cancelling generation.")
                                        while not llm_to_tts_queue.empty():
                                            llm_to_tts_queue.get_nowait()

                                        llm_to_tts_queue.put(terminate)
                                        full_response = ""
                                        await llm_model.cancel_dialogue_generation()
                                        break
                                except Exception: # Should be queue.Empty, but being safe
                                    pass # No interrupt message

                        #save to short term memory
                        if full_response:
                            if not continue_final_message:
                                conversation_history.append(prompt)
                            conversation_history.append(full_response)
                                    
                    
                
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in GPU runner loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        tts_consumer_task.cancel()
        try:
            await tts_consumer_task
        except asyncio.CancelledError:
            logger.info("TTS consumer task cancelled.")

    tts_model.cleanup()
    logger.info("GPU worker has shut down.")


def gpu_worker(shutdown_event: mp.Event, llm_control_queue: mp.Queue, llm_to_tts_queue: mp.Queue, tts_go_event: mp.Event):
    """Entry point for the GPU worker process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(gpu_runner(shutdown_event, llm_control_queue, llm_to_tts_queue, tts_go_event))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close() 