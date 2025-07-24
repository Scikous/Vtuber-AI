import multiprocessing as mp
from src.utils.app_utils import setup_project_root
from src.utils import logger as app_logger
from src.common import config as app_config
from src.utils.performance_utils import apply_system_optimizations, get_cuda_utilization
from STT_Wizard.STT import WhisperSTT
import torch
import time

def stt_client_worker(shutdown_event: mp.Event, user_has_stopped_speaking_event: mp.Event, stt_stream_queue: mp.Queue, gpu_request_queue: mp.Queue, worker_event: mp.Event):
    setup_project_root()
    logger = app_logger.get_logger("STTClientWorker")
    config = app_config.load_config()
    stt_settings = config.get("stt_settings", {})
    
    apply_system_optimizations(logger, use_cuda=True)
    stt_handler = WhisperSTT(**stt_settings)
    worker_id = "STT"

    # Model loading (low priority)    
    gpu_request_queue.put({"type": "acquire", "priority": 5, "worker_id": worker_id})
    worker_event.wait()
    try:
        if torch.cuda.is_available() and get_cuda_utilization() > 0.9:
            logger.warning("GPU memory usage high, delaying STT model loading.")
            while get_cuda_utilization() > 0.9:
                time.sleep(1)
        stt_handler._load_model()
    finally:
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})

    
    def sentence_callback(sentence):
        if sentence:
            logger.info(f"STT Client: Sending sentence: '{sentence}'")
            stt_stream_queue.put(sentence)
    try:
        stt_handler.listen_and_transcribe(sentence_callback, shutdown_event, user_has_stopped_speaking_event, gpu_request_queue=gpu_request_queue, gpu_request_event=worker_event, worker_id=worker_id)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in STT client worker.")
    
    logger.info("STT client worker has shut down.")