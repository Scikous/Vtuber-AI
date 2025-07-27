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

    # Load model with GPU management
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

    def transcription_with_gpu(audio_data, **kwargs):
            gpu_request_queue.put({"type": "acquire", "priority": 1, "worker_id": worker_id})
            worker_event.wait()
            try:
                return stt_handler.transcribe_audio_sync(audio_data, **kwargs)
            finally:
                gpu_request_queue.put({"type": "release", "worker_id": worker_id})

    def on_speech_start():
        user_has_stopped_speaking_event.clear()
        logger.info("Speech detected, clearing user_has_stopped_speaking_event")

    def on_speech_end():
        user_has_stopped_speaking_event.set()
        logger.info("End of speech detected, setting user_has_stopped_speaking_event")

    def sentence_callback(sentence):
        if sentence:
            logger.info(f"STT Client: Sending sentence: '{sentence}'")
            stt_stream_queue.put(sentence)

    try:
        logger.info("Starting STT client worker.")
        stt_handler.process_audio(
            sentence_callback=sentence_callback,
            transcription_func=transcription_with_gpu,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in STT client worker.")
    except Exception as e:
        logger.error(f"Error in STT client worker: {e}")

    logger.info("STT client worker has shut down.")