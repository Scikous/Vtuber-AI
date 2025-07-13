import multiprocessing as mp
import sys
import os
import time
import uuid
import threading

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.common import config as app_config
from STT_Wizard.STT import WhisperSTT

def stt_client_worker(shutdown_event: mp.Event, user_has_stopped_speaking_event: mp.Event, request_queue: mp.Queue, result_queue: mp.Queue):
    """
    Client worker for Speech-to-Text.
    - Uses STT_Wizard to capture audio and perform VAD.
    - Sends audio chunks to the GPU worker for transcription via a callback.
    """
    logger = app_logger.get_logger("STTClientWorker")
    config = app_config.load_config()
    stt_settings = config.get("stt_settings", {})
    
    # This instance will not load the model, it will only be used for audio utilities.
    stt_handler = WhisperSTT(**stt_settings)
    
    def send_full_utterance(audio_data_np):
        """Callback for the full audio utterance."""
        if audio_data_np.size == 0: return
        logger.info(f"Sending full utterance ({len(audio_data_np)/16000:.2f}s) for transcription...")
        request_id = str(uuid.uuid4())
        request_queue.put({
            "id": request_id,
            "task": "transcribe_full",
            "audio_data": audio_data_np
        })

    def send_fast_chunk(audio_data_np):
        """Callback for the fast, initial audio chunk."""
        if audio_data_np.size == 0: return
        logger.info(f"Sending fast chunk ({len(audio_data_np)/16000:.2f}s) for transcription...")
        request_id = str(uuid.uuid4())
        request_queue.put({
            "id": request_id,
            "task": "transcribe_fast",
            "audio_data": audio_data_np
        })

    # The listening process is blocking, so it runs in a separate thread.
    listen_thread = threading.Thread(
        target=stt_handler.listen_and_buffer,
        args=(send_full_utterance, send_fast_chunk, shutdown_event, user_has_stopped_speaking_event)
    )

    logger.info("STT client worker started.")
    listen_thread.start()
    
    try:
        # Keep the main worker process alive while the listening thread runs
        listen_thread.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in STT client worker.")
    except Exception as e:
        logger.error(f"An error occurred in the STT client worker: {e}", exc_info=True)
    
    logger.info("STT client worker has shut down.") 