import multiprocessing as mp
import sys
import os
import threading

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import logger as app_logger
from src.common import config as app_config
from STT_Wizard.STT import WhisperSTT

def stt_client_worker(shutdown_event: mp.Event, user_has_stopped_speaking_event: mp.Event, stt_stream_queue: mp.Queue):
    """
    Client worker for Speech-to-Text.
    - Uses STT_Wizard to capture audio, perform VAD, and transcribe sentences.
    - Puts transcribed sentences onto the stt_stream_queue.
    """
    logger = app_logger.get_logger("STTClientWorker")
    config = app_config.load_config()
    stt_settings = config.get("stt_settings", {})
    
    stt_handler = WhisperSTT(**stt_settings)
    stt_handler._load_model()

    # the service process is specific, NO self-defined callbacks
    def sentence_callback(sentence):
        """Callback to send each sentence to the context worker."""
        if sentence:
            logger.info(f"STT Client: Sending sentence: '{sentence}'")
            stt_stream_queue.put(sentence)

    # The listening process is blocking, so it runs in a separate thread.
    listen_thread = threading.Thread(
        target=stt_handler.listen_and_transcribe,
        args=(sentence_callback, shutdown_event, user_has_stopped_speaking_event)
    )

    logger.info("STT client worker started.")
    listen_thread.start()
    
    try:
        listen_thread.join()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received in STT client worker.")
    
    logger.info("STT client worker has shut down.") 