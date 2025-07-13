import multiprocessing as mp
import time
import signal
import sys
import os
from typing import Dict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common import config as app_config
from src.utils import logger as app_logger

# Import new worker functions
from src.workers.stt_client_worker import stt_client_worker
from src.workers.dialogue_client_worker import dialogue_client_worker
from src.workers.gpu_worker import gpu_worker

class ProcessOrchestrator:
    """
    Manages the lifecycle of all service processes under the new architecture.
    - GPU Worker: A single process that manages all models on the GPU.
    - Client Workers: CPU-bound processes for STT, Dialogue logic, etc.
    """
    def __init__(self):
        self.logger = app_logger.get_logger("ProcessOrchestrator")
        self.config = app_config.load_config()
        self.shutdown_event = mp.Event()
        self.user_has_stopped_speaking_event = mp.Event()
        
        # --- Inter-Process Communication Queues ---
        self.queues = {
            # GPU Worker Queues
            "stt_requests": mp.Queue(),
            "llm_requests": mp.Queue(),
            "tts_requests": mp.Queue(),

            # Client Worker Communication Queues
            "stt_to_llm_queue": mp.Queue(), # STT results posted here
            "llm_to_tts_queue": mp.Queue(), # LLM results posted here
        }

        # Worker process definitions
        self.workers: Dict[str, mp.Process] = {}
        self.worker_definitions = {
            "gpu": (gpu_worker, [self.shutdown_event, self.user_has_stopped_speaking_event, self.queues]),
            "stt_client": (stt_client_worker, [
                self.shutdown_event, 
                self.user_has_stopped_speaking_event,
                self.queues["stt_requests"], 
                self.queues["stt_to_llm_queue"]
            ]),
            "dialogue_client": (dialogue_client_worker, [
                self.shutdown_event,
                self.queues["stt_to_llm_queue"],
                self.queues["llm_requests"],
                self.queues["llm_to_tts_queue"],
                self.queues["tts_requests"]
            ]),
        }
        
        self.logger.info("ProcessOrchestrator initialized with new architecture.")

    def start_workers(self):
        """
        Starts all defined worker processes.
        """
        self.logger.info("Starting worker processes...")
        for name, (target, args) in self.worker_definitions.items():
            process = mp.Process(target=target, args=tuple(args), name=name)
            self.workers[name] = process
            process.start()
            self.logger.info(f"Started {name} worker with PID: {process.pid}")

    def stop_workers(self):
        """
        Stops all running worker processes gracefully.
        """
        self.logger.info("Stopping all worker processes by setting shutdown event...")
        self.shutdown_event.set()

        for name, process in self.workers.items():
            process.join(timeout=10) # Wait for graceful exit
            if process.is_alive():
                self.logger.warning(f"{name} worker (PID: {process.pid}) did not terminate gracefully. Forcing shutdown.")
                process.kill()
            else:
                self.logger.info(f"{name} worker (PID: {process.pid}) terminated with exit code {process.exitcode}.")

    def run(self):
        """
        Main run loop for the orchestrator.
        """
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        self.start_workers()
        
        try:
            # Keep the main process alive to monitor workers
            while not self.shutdown_event.is_set():
                for name, process in self.workers.items():
                    if not process.is_alive():
                        self.logger.error(f"Worker '{name}' has terminated unexpectedly with exit code {process.exitcode}. Shutting down all workers.")
                        self.shutdown_event.set()
                        break
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received in main orchestrator. Shutting down...")
        finally:
            self.stop_workers()
            self.logger.info("ProcessOrchestrator has shut down.")

    def _handle_signal(self, signum, frame):
        self.logger.info(f"Signal {signal.strsignal(signum)} received. Initiating graceful shutdown...")
        self.shutdown_event.set()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    orchestrator = ProcessOrchestrator()
    orchestrator.run() 