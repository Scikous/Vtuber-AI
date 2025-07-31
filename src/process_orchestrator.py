import multiprocessing as mp
import time
import signal

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #only temp, for testing
from typing import Dict
from src.utils.app_utils import setup_project_root
from src.utils import logger as app_logger
from src.common import config as app_config
from src.workers.gpu_manager_worker import gpu_manager
from src.workers.context_llm_worker import context_llm_worker
from src.workers.stt_client_worker import stt_client_worker
from src.workers.llm_worker import llm_worker
from src.workers.tts_worker import tts_worker


class ProcessOrchestrator:
    def __init__(self, llm_output_display_queue: mp.Queue=None, tts_mute_event: mp.Event=mp.Event(), stt_mute_event: mp.Event=mp.Event(), is_managed=False):
        setup_project_root()
        self.logger = app_logger.get_logger("ProcessOrchestrator")
        self.config = app_config.load_config()
        self.shutdown_event = mp.Event()
         # --- Passed from AppManager ---
        self.llm_output_display_queue = llm_output_display_queue
        self.tts_mute_event = tts_mute_event
        self.stt_mute_event = stt_mute_event
        self.is_managed = is_managed
        # ----------------------------
        self.user_has_stopped_speaking_event = mp.Event()
        self.gpu_ready_event = mp.Event()  # For startup synchronization
        self.max_gpu_slots = 2

        self.worker_events = {
                "STT": mp.Event(),
                "Context_LLM": mp.Event(),
                "LLM": mp.Event(),
                "TTS": mp.Event()
            }
        # Inter-Process Communication Queues with limits -- helps further avoid GPU saturation
        queue_settings = self.config.get("queue_settings", {})
        self.queues = {
            "stt_stream_queue": mp.Queue(maxsize=queue_settings.get("stt_stream_queue_maxsize", 3)),
            "llm_control_queue": mp.Queue(maxsize=queue_settings.get("llm_control_queue_maxsize", 3)),
            "llm_to_tts_queue": mp.Queue(maxsize=queue_settings.get("llm_to_tts_queue_maxsize", 3)),
            "gpu_request_queue": mp.Queue(maxsize=queue_settings.get("gpu_request_queue_maxsize", 0)),
        }

        self.workers: Dict[str, mp.Process] = {}

        # Update worker_definitions to pass gpu_request_queue
        self.worker_definitions = {
            "gpu_manager": (gpu_manager, [
                self.queues["gpu_request_queue"],
                self.worker_events,
                self.max_gpu_slots
            ]),
            "stt_client": (stt_client_worker, [
                self.shutdown_event,
                self.user_has_stopped_speaking_event,
                self.queues["stt_stream_queue"],
                self.queues["gpu_request_queue"],
                self.worker_events["STT"],
                self.stt_mute_event,

            ]),
            "context_llm": (context_llm_worker, [
                self.shutdown_event,
                self.queues["stt_stream_queue"],
                self.queues["llm_control_queue"],
                self.user_has_stopped_speaking_event,
                self.gpu_ready_event,
                self.queues["gpu_request_queue"],
                self.worker_events["Context_LLM"],
            ]),
            "llm": (llm_worker, [
                self.shutdown_event,
                self.queues["llm_control_queue"],
                self.queues["llm_to_tts_queue"],
                self.gpu_ready_event,
                self.queues["gpu_request_queue"],
                self.worker_events["LLM"],
                self.llm_output_display_queue,

            ]),
            "tts": (tts_worker, [
                self.shutdown_event,
                self.queues["llm_to_tts_queue"],
                self.user_has_stopped_speaking_event,
                self.queues["gpu_request_queue"],
                self.worker_events["TTS"],
                self.tts_mute_event,
            ]),
        }
        self.logger.info("ProcessOrchestrator initialized with priority-based GPU management.")


    def start_workers(self):
        self.logger.info("Starting worker processes...")
        for name, (target, args) in self.worker_definitions.items():
            process = mp.Process(target=target, args=tuple(args), name=name)
            self.workers[name] = process
            process.start()
            self.logger.info(f"Started {name} worker with PID: {process.pid}")

    def stop_workers(self):
        self.logger.info("Stopping all worker processes...")
        self.shutdown_event.set()
        for name, process in self.workers.items():
            process.join(timeout=10)
            if process.is_alive():
                self.logger.warning(f"{name} worker (PID: {process.pid}) did not terminate gracefully. Forcing shutdown.")
                process.kill()
            else:
                self.logger.info(f"{name} worker (PID: {process.pid}) terminated with exit code {process.exitcode}.")

    def run(self):
        if not self.is_managed:
            self.logger.info("Running in standalone mode. Registering signal handlers.")
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        else:
            self.logger.info("Running in managed mode. Skipping signal handler registration.")
        self.start_workers()
        try:
            while not self.shutdown_event.is_set():
                for name, process in self.workers.items():
                    if not process.is_alive():
                        self.logger.error(f"Worker '{name}' terminated unexpectedly with exit code {process.exitcode}. Shutting down.")
                        self.shutdown_event.set()
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            self.stop_workers()
            self.logger.info("ProcessOrchestrator has shut down.")

    def _handle_signal(self, signum, frame):
        self.logger.info(f"Signal {signal.strsignal(signum)} received. Initiating shutdown...")
        self.shutdown_event.set()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    orchestrator = ProcessOrchestrator()
    orchestrator.run()