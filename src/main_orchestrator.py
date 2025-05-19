"""
Main Orchestrator for Vtuber-AI
This module coordinates the core workflow, integrating services and utilities.
"""
import asyncio
import logging
import os
import multiprocessing
from multiprocessing import Queue as MPQueue # Renamed to avoid conflict with asyncio.Queue
from collections import deque
from dotenv import load_dotenv

from services.service_manager import ServiceManager
from utils import logger as app_logger
from utils.file_operations import write_messages_csv
from utils.env_utils import get_env_var
# from utils.app_utils import change_dir # change_dir might be used within specific services
from common import config as app_config
from common import queues
from utils.live_chat_process_handler import live_chat_process_target # For the separate live chat process

# Temporary direct imports for LLM and other components not yet refactored into src/services
# These will be replaced by service-specific imports later.
# Ensure these top-level modules (LLM, voiceAI, livechatAPI) are in PYTHONPATH or accessible.
# This might require adjusting sys.path in run.py or ensuring the execution context is the project root.
import sys
# Example: Add project root to sys.path if not already there. This is often handled by the entry point (run.py)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)


try:
    from LLM_Wizard.models import VtuberExllamav2
    from LLM_Wizard.model_utils import LLMUtils
    from LLM_Wizard.llm_templates import PromptTemplate as LLMPromptTemplate # Renamed to avoid conflict
    # Imports for STT, TTS, LiveChat will be handled by their respective services
    # from TTS_Wizard.GPT_Test.tts_exp import send_tts_request, run_playback_thread, tts_queue as tts_output_queue_brain
    # from STT_Wizard.STT import speech_to_text # Assuming STT_Wizard/STT.py exists and has speech_to_text
    # from Livechat_Wizard.livechat import LiveChatController # Assuming Livechat_Wizard/livechat.py exists and has LiveChatController
except ImportError as e:
    print(f"Critical Import Error: Could not import core AI modules (LLM, etc.): {e}. Ensure PYTHONPATH is set correctly or modules are accessible.")
    VtuberExllamav2, LLMUtils, LLMPromptTemplate = None, None, None # Graceful degradation or error handling needed
import time
class MainOrchestrator:
    live_chat_process = None # Class attribute to hold the live chat process
    def __init__(self):
        self.logger = app_logger.get_logger("MainOrchestrator")
        self.logger.info("Initializing MainOrchestrator...")
        load_dotenv() # Load .env file variables into environment

        self.config = app_config.load_config()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Initialize shared queues
        self.speech_queue = queues.get_speech_queue()
        self.live_chat_queue = queues.get_live_chat_queue()
        self.llm_output_queue = queues.get_llm_output_queue()
        self.audio_output_queue = queues.get_audio_output_queue()
        self.mp_live_chat_message_queue = queues.get_mp_live_chat_message_queue()

        self.character_name = None
        self.user_name = None
        self.llm_prompt_template = None
        self.character_model = None
        # LLM loading will be done in run_async_loop

        self.naive_short_term_memory = deque(maxlen=self.config.get("short_term_memory_maxlen", 6))
        self.speaker_name = self.config.get("speaker_name", "_") # Default speaker name

        self.conversation_log_file = get_env_var("CONVERSATION_LOG_FILE")
        if self.conversation_log_file and not os.path.isabs(self.conversation_log_file):
            self.conversation_log_file = os.path.join(self.project_root, self.conversation_log_file)
        
        self.write_to_log_fn = None
        if self.conversation_log_file:
            self.write_to_log_fn = write_messages_csv
            self.logger.info(f"Conversation logging enabled to: {self.conversation_log_file}")
        else:
            self.logger.info("Conversation logging is disabled.")

        # Prepare shared resources for services
        self.shared_resources = {
            "config": self.config,
            "logger": self.logger,
            "queues": {
                "speech_queue": self.speech_queue,
                "live_chat_queue": self.live_chat_queue,
                "llm_output_queue": self.llm_output_queue,
                "audio_output_queue": self.audio_output_queue,
                "mp_live_chat_message_queue": self.mp_live_chat_message_queue,
                # Add other queues like tts_input_queue if they become shared
            },
            "character_model": self.character_model,
            "llm_prompt_template": self.llm_prompt_template,
            "naive_short_term_memory": self.naive_short_term_memory,
            "character_name": self.character_name,
            "user_name": self.user_name,
            "speaker_name": self.speaker_name,
            "conversation_log_file": self.conversation_log_file,
            "write_to_log_fn": self.write_to_log_fn,
            "project_root": self.project_root
            # LLM specific resources will be added in run_async_loop
        }

        # ServiceManager and services will be initialized in run_async_loop after LLM loading
        self.service_manager = None 

        self.logger.info("MainOrchestrator initialized (LLM and services will be set up in run_async_loop).")

    async def load_character_and_llm(self):
        self.logger.info("Loading character and LLM...")
        if not LLMUtils or not LLMPromptTemplate or not VtuberExllamav2:
            self.logger.error("LLM utilities or models not imported. Cannot load character/LLM.")
            # Potentially raise an error or handle this state appropriately
            return

        character_info_json_path = self.config.get("character_info_json", "LLM_Wizard/characters/character.json")
        if not os.path.isabs(character_info_json_path):
            character_info_json_path = os.path.join(self.project_root, character_info_json_path)
        
        if not os.path.exists(character_info_json_path):
            self.logger.error(f"Character info JSON not found at: {character_info_json_path}")
            # Handle error: maybe load defaults or raise exception
            return

        instructions, self.user_name, self.character_name = LLMUtils.load_character(character_info_json_path)
        self.llm_prompt_template = LLMPromptTemplate(instructions, self.user_name, self.character_name)
        
        # For Exllamav2 model; configuration for other models would differ
        # Model path/config should ideally come from self.config
        self.character_model = VtuberExllamav2.load_model_exllamav2(character_name=self.character_name)
        self.logger.info(f"Character '{self.character_name}' and LLM loaded.")

    def register_services(self):
        self.logger.info("Registering services...")
        # Import service classes
        from services.stt_service import STTService
        from services.tts_service import TTSService
        from services.dialogue_service import DialogueService
        from services.live_chat_service import LiveChatService
        from services.audio_stream_service import AudioStreamService

        # Instantiate and register services with self.shared_resources
        self.service_manager.register_service(STTService(self.shared_resources))
        self.service_manager.register_service(DialogueService(self.shared_resources))
        self.service_manager.register_service(TTSService(self.shared_resources))
        self.service_manager.register_service(LiveChatService(self.shared_resources))
        self.service_manager.register_service(AudioStreamService(self.shared_resources))
        self.logger.info("All services registered.")

    async def run_async_loop(self):
        self.logger.info("Starting asynchronous event loop and workers...")

        # Load character and LLM model first
        await self.load_character_and_llm()

        # Update shared_resources with loaded LLM components
        self.shared_resources["character_model"] = self.character_model
        self.shared_resources["llm_prompt_template"] = self.llm_prompt_template
        self.shared_resources["character_name"] = self.character_name
        self.shared_resources["user_name"] = self.user_name

        # Now initialize ServiceManager and register services
        self.service_manager = ServiceManager(self.shared_resources)
        self.register_services()

        # Start the separate live chat process if the queue is available
        if self.mp_live_chat_message_queue:
            self.logger.info("Starting live chat process...")
            try:
                MainOrchestrator.live_chat_process = multiprocessing.Process(
                    target=live_chat_process_target, 
                    args=(self.mp_live_chat_message_queue,)
                )
                MainOrchestrator.live_chat_process.daemon = True # Ensure it exits when main process exits
                MainOrchestrator.live_chat_process.start()
                self.logger.info(f"Live chat process started with PID: {MainOrchestrator.live_chat_process.pid}")
            except Exception as e:
                self.logger.error(f"Failed to start live chat process: {e}", exc_info=True)
                MainOrchestrator.live_chat_process = None # Ensure it's None if start fails
        else:
            self.logger.warning("mp_live_chat_message_queue not available, live chat process will not be started.")
        
        # Start all registered async services
        if self.service_manager:
            await self.service_manager.start_all_services()
        else:
            self.logger.error("ServiceManager not initialized, cannot start services.")
            return # Critical error, cannot proceed

        try:
            # Keep the main orchestrator alive, services run in background
            # Or, if services are tasks, await them here
            # For now, we assume services manage their own lifecycle once started
            while True: 
                await asyncio.sleep(3600) # Keep alive, or handle graceful shutdown signals
        except asyncio.CancelledError:
            self.logger.info("Main orchestrator async loop cancelled.")
        finally:
            self.logger.info("Stopping all services...")
            await self.service_manager.stop_all_services()
            self.logger.info("All async services stopped.")

            # Terminate the live chat process
            if MainOrchestrator.live_chat_process and MainOrchestrator.live_chat_process.is_alive():
                self.logger.info(f"Terminating live chat process (PID: {MainOrchestrator.live_chat_process.pid})...")
                try:
                    MainOrchestrator.live_chat_process.terminate() # Send SIGTERM
                    MainOrchestrator.live_chat_process.join(timeout=5) # Wait for graceful shutdown
                    if MainOrchestrator.live_chat_process.is_alive():
                        self.logger.warning(f"Live chat process (PID: {MainOrchestrator.live_chat_process.pid}) did not terminate gracefully, attempting to kill.")
                        MainOrchestrator.live_chat_process.kill() # Send SIGKILL
                        MainOrchestrator.live_chat_process.join(timeout=2)
                    if MainOrchestrator.live_chat_process.is_alive():
                        self.logger.error(f"Live chat process (PID: {MainOrchestrator.live_chat_process.pid}) could not be stopped.")
                    else:
                        self.logger.info(f"Live chat process (PID: {MainOrchestrator.live_chat_process.pid}) stopped.")
                except Exception as e:
                    self.logger.error(f"Error during live chat process termination: {e}", exc_info=True)
            elif MainOrchestrator.live_chat_process:
                self.logger.info("Live chat process was not alive or already stopped.")
            else:
                self.logger.info("No live chat process was started.")

    def run(self):
        self.logger.info("Starting Vtuber-AI Orchestration...")
        try:
            asyncio.run(self.run_async_loop())
        except KeyboardInterrupt:
            self.logger.info("Orchestration interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            self.logger.error(f"Unhandled exception in MainOrchestrator run: {e}", exc_info=True)
        finally:
            self.logger.info("Vtuber-AI Orchestration shutting down.")

if __name__ == "__main__":
    # Basic logging setup for standalone execution
    # In a real application, run.py would handle this more robustly.
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    
    # This is primarily for testing the orchestrator directly.
    # The main entry point will be run.py in the project root.
    orchestrator = MainOrchestrator()
    orchestrator.run()