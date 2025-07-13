import multiprocessing as mp
import sys
import os
import time
import threading
import asyncio

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations
from src.common import config as app_config

# Model Imports
from STT_Wizard.STT import WhisperSTT
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from LLM_Wizard.model_utils import load_character, contains_sentence_terminator
from TTS_Wizard.realtimetts import RealTimeTTS, pipertts_engine, coquitts_engine

def llm_thread_worker(shutdown_event, request_queue, response_queue, logger, config, ready_event):
    """Dedicated thread for loading and running async LLM tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def llm_runner():
        llm_settings = config.get("llm_settings", {})
        character_info_json_path = llm_settings.get("character_info_json", "LLM_Wizard/characters/character.json")
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
        
        logger.info("LLM Thread: Loading model...")
        async with await VtuberExllamav2.load_model(config=model_config) as llm_model:
            logger.info("✅ LLM Thread: Model loaded.")
            ready_event.set() # Signal that the LLM is ready
            
            while not shutdown_event.is_set():
                if not request_queue.empty():
                    request = request_queue.get()
                    logger.debug(f"LLM thread got request: {str(request)[:100]}")

                    min_sentence_len = config.get("min_sentence_len", 8)
                    async_job = await llm_model.dialogue_generator(
                        request['prompt'],
                        conversation_history=request.get('history'),
                        max_tokens=request.get('max_tokens', 512)
                    )

                    transcribe_type = request.get('task', "")
                    tts_buffer = ""
                    async for result in async_job:
                        if shutdown_event.is_set():
                            await llm_model.cancel_dialogue_generation()
                            break
                        chunk_text = result.get("text", "")
                        if chunk_text:
                            tts_buffer += chunk_text
                            if contains_sentence_terminator(chunk_text):
                                text_to_send = tts_buffer.strip()
                                if text_to_send and len(text_to_send) >= min_sentence_len:
                                    response_queue.put({'id': request['id'], 'text': text_to_send, 'is_final': False})
                                    tts_buffer = ""
                                    if transcribe_type == "transcribe_fast":
                                        await llm_model.cancel_dialogue_generation()
                                        break
                    
                    if tts_buffer.strip():
                        response_queue.put({'id': request['id'], 'text': tts_buffer.strip(), 'is_final': True})
                else:
                    await asyncio.sleep(0.05)
    
    loop.run_until_complete(llm_runner())
    logger.info("LLM thread has shut down.")

def tts_thread_worker(shutdown_event, request_queue, logger, config, user_has_stopped_speaking_event, ready_event):
    """Dedicated thread for loading and running async TTS tasks."""
    tts_settings = config.get("tts_settings", {})
    tts_engine_name = tts_settings.get("tts_engine", "piper")
    if tts_engine_name == "piper":
        model_file = tts_settings.get("model_file")
        config_file = tts_settings.get("config_file")
        piper_path = tts_settings.get("piper_path")
        tts_engine = pipertts_engine(model_file, config_file, piper_path)
    elif tts_engine_name == "coqui":
        tts_engine = coquitts_engine(use_deepspeed=True)
    
    tts_model = RealTimeTTS(tts_engine, user_has_stopped_speaking_event=user_has_stopped_speaking_event)
    logger.info("✅ TTS Thread: Model loaded.")
    ready_event.set() # Signal that TTS is ready

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def process_requests():
        while not shutdown_event.is_set():
            if not request_queue.empty():
                request = request_queue.get()
                logger.debug(f"TTS thread got request: {str(request)[:100]}")
                await tts_model.tts_request_async(request['text'])
            else:
                await asyncio.sleep(0.05)

    loop.run_until_complete(process_requests())
    tts_model.cleanup()
    logger.info("TTS thread has shut down.")


def gpu_worker(shutdown_event: mp.Event, user_has_stopped_speaking_event: mp.Event, queues: dict):
    logger = app_logger.get_logger("GPUWorker")
    config = app_config.load_config()
    apply_system_optimizations(logger)

    logger.info("Loading synchronous STT model...")
    stt_settings = config.get("stt_settings", {})
    stt_model = WhisperSTT(**stt_settings)
    stt_model._load_model()
    logger.info("✅ STT model loaded.")

    # Events to signal when async models are ready
    llm_ready_event = threading.Event()
    tts_ready_event = threading.Event()

    # Start dedicated threads for async models
    shutdown_threads_event = threading.Event()
    
    llm_thread = threading.Thread(target=llm_thread_worker, args=(shutdown_threads_event, queues["llm_requests"], queues["llm_to_tts_queue"], logger, config, llm_ready_event))
    tts_thread = threading.Thread(target=tts_thread_worker, args=(shutdown_threads_event, queues["tts_requests"], logger, config, user_has_stopped_speaking_event, tts_ready_event))
    
    llm_thread.start()
    tts_thread.start()
    
    # Wait for models to become ready before processing any requests
    logger.info("GPU Worker: Waiting for models to load...")
    llm_ready_event.wait()
    tts_ready_event.wait()
    logger.info("GPU Worker: All models loaded. Starting processing loop.")

    while not shutdown_event.is_set():
        try:
            if not queues["stt_requests"].empty():
                request = queues["stt_requests"].get()
                task = request.get("task")
                logger.debug(f"Processing STT request of type: {task}...")

                if task == "transcribe_fast":
                    text = stt_model.transcribe_audio_sync(request['audio_data'], beam_size=1, temperature=0.6)
                    queues["stt_to_llm_queue"].put({'id': request['id'], 'text': text, 'type': 'fast'})
                
                elif task == "transcribe_full":
                    text = stt_model.transcribe_audio_sync(request['audio_data'], beam_size=5, temperature=0.0)
                    queues["stt_to_llm_queue"].put({'id': request['id'], 'text': text, 'type': 'full'})
            else:
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"An error occurred in the GPU worker main loop: {e}", exc_info=True)
            time.sleep(1)

    logger.info("Shutdown signal received. Stopping threads and releasing resources.")
    shutdown_threads_event.set()
    llm_thread.join()
    tts_thread.join()
    
    logger.info("GPU worker has shut down.") 