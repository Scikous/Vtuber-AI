import multiprocessing as mp
import asyncio
from src.utils.env_utils import setup_project_root
from src.utils import logger as app_logger
from src.utils.performance_utils import apply_system_optimizations, async_check_gpu_memory
from src.common import config as app_config
from TTS_Wizard.realtimetts import RealTimeTTS, pipertts_engine, coquitts_engine

app_logger.setup_logging()
logger = app_logger.get_logger("TTSWorker")

async def tts_runner(shutdown_event, llm_to_tts_queue, 
                     user_has_stopped_speaking_event, gpu_request_queue, worker_event, worker_id="TTS", tts_mute_event=None):
    config = app_config.load_config()
    apply_system_optimizations(logger, use_cuda=config.get("tts_settings", {}).get("use_cuda", True))
    
    tts_settings = config.get("tts_settings", {})
    tts_engine_name = tts_settings.get("tts_engine", "piper")
    TERMINATE_OUTPUT = config.get("TERMINATE_OUTPUT", "w1zt3r")
    gpu_request_queue.put({"type": "acquire", "priority": 1, "worker_id":  worker_id})
    worker_event.wait()
    try:
        if tts_engine_name == "piper":
            tts_engine = pipertts_engine(tts_settings.get("model_file"), tts_settings.get("config_file"), tts_settings.get("piper_path"))
        else:
            tts_engine = coquitts_engine(use_deepspeed=tts_settings.get("use_deepspeed", True))
        voice_to_clone = tts_settings.get("voice_to_clone_file", None)#implement this dumbo
        tts_model = RealTimeTTS(tts_engine, tts_playback_approved_event=user_has_stopped_speaking_event)
        logger.info("✅ TTS model loaded.")
    finally:
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})


    while not shutdown_event.is_set():
        try:
            #control panel or keyboard can mute
            if tts_mute_event.is_set():
                await tts_model.tts_request_clear()
                continue

            sentence = await asyncio.to_thread(llm_to_tts_queue.get, timeout=0.1)
            if sentence == TERMINATE_OUTPUT:
                await tts_model.stop()
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
                    if not tts_mute_event.is_set():
                        await tts_model.speak(sentence)
                finally:
                    gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        except mp.queues.Empty:
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Error in TTS runner: {e}", exc_info=True)
    
    tts_model.shutdown()

def tts_worker(shutdown_event: mp.Event, llm_to_tts_queue: mp.Queue, user_has_stopped_speaking_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event, tts_mute_event: mp.Event):
    setup_project_root()
    worker_id = "TTS"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tts_runner(shutdown_event, llm_to_tts_queue, user_has_stopped_speaking_event, gpu_request_queue, worker_event, worker_id, tts_mute_event))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("TTS worker has shut down.")