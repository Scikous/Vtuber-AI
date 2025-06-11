#!/usr/bin/env python3
"""
Multiprocessing Orchestrator for Vtuber-AI
This module coordinates the core workflow using separate processes for STT, TTS, and LLM components.
Designed for significantly better performance than the async-based system.
"""
import multiprocessing as mp
import logging
import os
import time
import signal
import sys
from multiprocessing import Queue, Process, Event, Value
from ctypes import c_bool
from dotenv import load_dotenv

from utils import logger as app_logger
from utils.file_operations import write_messages_csv
from utils.env_utils import get_env_var
from common import config as app_config

# Import process worker functions
from multiprocess_workers.stt_worker import stt_process_worker
from multiprocess_workers.llm_worker import llm_process_worker
from multiprocess_workers.tts_worker import tts_process_worker
from multiprocess_workers.audio_worker import audio_process_worker
from multiprocess_workers.livechat_worker import livechat_process_worker

class MultiprocessOrchestrator:
    def __init__(self):
        self.logger = app_logger.get_logger("MultiprocessOrchestrator")
        self.logger.info("Initializing MultiprocessOrchestrator...")
        load_dotenv()  # Load .env file variables into environment

        self.config = app_config.load_config()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Multiprocessing queues for inter-process communication
        # Using smaller maxsize for better memory management and faster processing
        self.speech_queue = Queue(maxsize=1)  # STT -> LLM
        self.live_chat_queue = Queue(maxsize=1)  # LiveChat -> LLM
        self.llm_output_queue = Queue()  # LLM -> TTS
        self.audio_output_queue = Queue()  # TTS -> Audio Player
        
        # Shared events for process coordination
        self.terminate_event = Event()
        self.terminate_current_dialogue_event = Event()
        self.is_audio_streaming_event = Event()
        self.immediate_livechat_fetch_event = Event()
        
        # Shared values for process communication
        self.audio_playing = Value(c_bool, False)
        
        # Process references
        self.processes = {}
        
        # Configuration for workers
        self.character_name = None
        self.user_name = None
        self.speaker_name = self.config.get("speaker_name", "User")
        
        self.conversation_log_file = get_env_var("CONVERSATION_LOG_FILE")
        if self.conversation_log_file and not os.path.isabs(self.conversation_log_file):
            self.conversation_log_file = os.path.join(self.project_root, self.conversation_log_file)
        
        # Prepare shared configuration for all processes
        self.shared_config = {
            "config": self.config,
            "project_root": self.project_root,
            "speaker_name": self.speaker_name,
            "conversation_log_file": self.conversation_log_file,
            "character_name": self.character_name,
            "user_name": self.user_name,
        }
        
        self.logger.info("MultiprocessOrchestrator initialized.")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
            self.terminate_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_stt_process(self):
        """Start the STT (Speech-to-Text) process."""
        self.logger.info("Starting STT process...")
        process = Process(
            target=stt_process_worker,
            args=(
                self.speech_queue,
                self.terminate_event,
                self.terminate_current_dialogue_event,
                self.is_audio_streaming_event,
                self.shared_config
            ),
            name="STT-Process"
        )
        process.start()
        self.processes["stt"] = process
        self.logger.info(f"STT process started with PID: {process.pid}")
    
    def start_llm_process(self):
        """Start the LLM (Large Language Model) process."""
        self.logger.info("Starting LLM process...")
        process = Process(
            target=llm_process_worker,
            args=(
                self.speech_queue,
                self.live_chat_queue,
                self.llm_output_queue,
                self.terminate_event,
                self.terminate_current_dialogue_event,
                self.shared_config
            ),
            name="LLM-Process"
        )
        process.start()
        self.processes["llm"] = process
        self.logger.info(f"LLM process started with PID: {process.pid}")
    
    def start_tts_process(self):
        """Start the TTS (Text-to-Speech) process."""
        self.logger.info("Starting TTS process...")
        process = Process(
            target=tts_process_worker,
            args=(
                self.llm_output_queue,
                self.audio_output_queue,
                self.terminate_event,
                self.terminate_current_dialogue_event,
                self.shared_config
            ),
            name="TTS-Process"
        )
        process.start()
        self.processes["tts"] = process
        self.logger.info(f"TTS process started with PID: {process.pid}")
    
    def start_audio_process(self):
        """Start the Audio playback process."""
        self.logger.info("Starting Audio process...")
        process = Process(
            target=audio_process_worker,
            args=(
                self.audio_output_queue,
                self.terminate_event,
                self.terminate_current_dialogue_event,
                self.audio_playing,
                self.is_audio_streaming_event,
                self.shared_config
            ),
            name="Audio-Process"
        )
        process.start()
        self.processes["audio"] = process
        self.logger.info(f"Audio process started with PID: {process.pid}")
    
    def start_livechat_process(self):
        """Start the LiveChat process."""
        self.logger.info("Starting LiveChat process...")
        process = Process(
            target=livechat_process_worker,
            args=(
                self.live_chat_queue,
                self.terminate_event,
                self.immediate_livechat_fetch_event,
                self.shared_config
            ),
            name="LiveChat-Process"
        )
        process.start()
        self.processes["livechat"] = process
        self.logger.info(f"LiveChat process started with PID: {process.pid}")
    
    def start_all_processes(self):
        """Start all worker processes."""
        self.logger.info("Starting all worker processes...")
        
        # Start processes in optimal order
        # self.start_audio_process()  # Start audio first as it's the final consumer
        self.start_tts_process()    # Start TTS before LLM
        self.start_llm_process()    # Start LLM before input processes
        self.start_stt_process()    # Start STT
        self.start_livechat_process()  # Start LiveChat
        
        self.logger.info("All worker processes started.")
    
    def monitor_processes(self):
        """Monitor all processes and restart if necessary."""
        while not self.terminate_event.is_set():
            for name, process in self.processes.items():
                if not process.is_alive():
                    self.logger.warning(f"Process {name} (PID: {process.pid}) has died. Exit code: {process.exitcode}")
                    # Optionally restart the process here
                    # For now, we'll just log and continue
            
            time.sleep(1)  # Check every second
    
    def stop_all_processes(self):
        """Stop all worker processes gracefully."""
        self.logger.info("Stopping all worker processes...")
        
        # Set termination event
        self.terminate_event.set()
        
        # Wait for processes to terminate gracefully
        for name, process in self.processes.items():
            if process.is_alive():
                self.logger.info(f"Waiting for {name} process (PID: {process.pid}) to terminate...")
                process.join(timeout=5)
                
                if process.is_alive():
                    self.logger.warning(f"Force terminating {name} process (PID: {process.pid})...")
                    process.terminate()
                    process.join(timeout=2)
                    
                    if process.is_alive():
                        self.logger.error(f"Force killing {name} process (PID: {process.pid})...")
                        process.kill()
                        process.join()
                
                self.logger.info(f"{name} process stopped.")
        
        self.logger.info("All worker processes stopped.")
    
    def run(self):
        """Main execution method."""
        self.logger.info("Starting Vtuber-AI Multiprocessing Orchestration...")
        
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Start all processes
            self.start_all_processes()
            
            # Monitor processes
            self.monitor_processes()
            
        except KeyboardInterrupt:
            self.logger.info("Orchestration interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            self.logger.error(f"Unhandled exception in MultiprocessOrchestrator run: {e}", exc_info=True)
        finally:
            self.stop_all_processes()
            self.logger.info("Vtuber-AI Multiprocessing Orchestration shutting down.")

if __name__ == "__main__":
    # Basic logging setup for standalone execution
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    )
    
    # Set multiprocessing start method
    if sys.platform == "win32":
        mp.set_start_method('spawn', force=True)
    else:
        mp.set_start_method('fork', force=True)
    
    orchestrator = MultiprocessOrchestrator()
    orchestrator.run()