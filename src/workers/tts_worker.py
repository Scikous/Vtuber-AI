import multiprocessing as mp
import asyncio
from src.utils.app_utils import setup_project_root
from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations, async_check_gpu_memory
from src.common import config as app_config
from TTS_Wizard.realtimetts import RealTimeTTS, pipertts_engine, coquitts_engine

async def tts_runner(shutdown_event, llm_to_tts_queue, 
                     user_has_stopped_speaking_event, gpu_request_queue, worker_event, worker_id="TTS"):
    logger = app_logger.get_logger("TTSWorker")
    config = app_config.load_config()
    apply_system_optimizations(logger, use_cuda=config.get("tts_settings", {}).get("use_cuda", True))
    
    tts_settings = config.get("tts_settings", {})
    tts_engine_name = tts_settings.get("tts_engine", "piper")
    TERMINATE_OUTPUT = config.get("TERMINATE_OUTPUT", "w1zt3r")
    # Model loading (low priority)
    gpu_request_queue.put({"type": "acquire", "priority": 5, "worker_id":  worker_id})
    worker_event.wait()
    try:
        if tts_engine_name == "piper":
            tts_engine = pipertts_engine(tts_settings.get("model_file"), tts_settings.get("config_file"), tts_settings.get("piper_path"))
        else:
            tts_engine = coquitts_engine(use_deepspeed=tts_settings.get("use_deepspeed", True))
        tts_model = RealTimeTTS(tts_engine, user_has_stopped_speaking_event=user_has_stopped_speaking_event)
        logger.info("✅ TTS model loaded.")
    finally:
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})


    while not shutdown_event.is_set():
        try:
            sentence = await asyncio.to_thread(llm_to_tts_queue.get)
            if sentence == TERMINATE_OUTPUT:
                await tts_model.tts_request_clear()
                continue
            if sentence:
                if user_has_stopped_speaking_event.is_set():
                    gpu_request_queue.put({"type": "acquire", "priority": 1, "worker_id":  worker_id})
                else:
                    gpu_request_queue.put({"type": "acquire", "priority": 4, "worker_id":  worker_id})
                worker_event.wait()
                logger.info(f"TTS Worker: Processing sentence: '{sentence}'")
                try:
                    await async_check_gpu_memory(logger)
                    await tts_model.tts_request_async(sentence)
                finally:
                    gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        except mp.queues.Empty:
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Error in TTS runner: {e}", exc_info=True)
    
    tts_model.cleanup()
    logger.info("TTS worker has shut down.")

def tts_worker(shutdown_event: mp.Event, llm_to_tts_queue: mp.Queue, user_has_stopped_speaking_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event):
    setup_project_root()
    worker_id = "TTS"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tts_runner(shutdown_event, llm_to_tts_queue, user_has_stopped_speaking_event, gpu_request_queue, worker_event, worker_id))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()