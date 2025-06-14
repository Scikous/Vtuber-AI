"""Main Orchestrator for Vtuber-AI
This module coordinates the core workflow with minimal responsibilities.
Services now handle their own model loading and configuration.
"""

import sys
import asyncio
import logging
import os
from dotenv import load_dotenv

from services.service_manager import ServiceManager
from utils import logger as app_logger
from common import config as app_config
from common import queues
class MainOrchestrator:
    live_chat_process = None # Class attribute to hold the live chat process
    
    def __init__(self):
        self.logger = app_logger.get_logger("MainOrchestrator")
        self.logger.info("Initializing MainOrchestrator...")
        load_dotenv() # Load .env file variables into environment

        self.config = app_config.load_config()
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, self.project_root) # Adds project root to path, so that Python will stop whining about module not found
        # Initialize shared queues - these are the core communication channels
        self.queues = {
            "speech_queue": queues.get_speech_queue(),
            "live_chat_queue": queues.get_live_chat_queue(),
            "llm_output_queue": queues.get_llm_output_queue(),
            "audio_output_queue": queues.get_audio_output_queue()
        }
        
        # Shared events for coordination between services
        self.shared_events = {
            "terminate_current_dialogue_event": asyncio.Event(),
            "is_audio_streaming_event": asyncio.Event(),
            "immediate_livechat_fetch_event": asyncio.Event()
        }
        
        # Minimal shared resources - only what truly needs to be shared
        self.shared_resources = {
            "config": self.config,
            "logger": self.logger,
            "queues": self.queues,
            "project_root": self.project_root,
            **self.shared_events
        }

        self.service_manager = None
        self.logger.info("MainOrchestrator initialized with minimal shared resources.")

    def register_services(self):
        """Register services with the service manager.
        Services now handle their own model loading and configuration.
        """
        self.logger.info("Registering services...")

        # Import service classes
        from services.stt_service import STTService
        from services.tts_service import TTSService
        from services.dialogue_service import DialogueService
        from services.live_chat_service import LiveChatService
        from services.audio_stream_service import AudioStreamService

        # Services are responsible for their own initialization
        # They receive only the minimal shared resources they actually need
        self.service_manager.register_service(STTService(self.shared_resources))
        self.service_manager.register_service(DialogueService(self.shared_resources))
        self.service_manager.register_service(TTSService(self.shared_resources))
        self.service_manager.register_service(LiveChatService(self.shared_resources))
        # self.service_manager.register_service(AudioStreamService(self.shared_resources))
        
        self.logger.info("All services registered.")

    async def run_async_loop(self):
        """Main orchestration loop with minimal responsibilities.
        Services handle their own model loading and lifecycle management.
        """
        self.logger.info("Starting asynchronous event loop...")

        # Initialize ServiceManager and register services
        self.service_manager = ServiceManager(self.shared_resources)
        self.register_services()

        # Start all registered services
        try:
            await self.service_manager.start_all_services()
            self.logger.info("All services started successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            return

        try:
            # Keep the orchestrator alive while services run
            while True: 
                await asyncio.sleep(3600) # Keep alive, or handle graceful shutdown signals
        except asyncio.CancelledError:
            self.logger.info("Main orchestrator async loop cancelled.")
        finally:
            await self._cleanup()
    
    async def _cleanup(self):
        """Handle cleanup of orchestrator resources."""
        self.logger.info("Starting orchestrator cleanup...")
        
        # Stop all services - they handle their own cleanup including models
        if self.service_manager:
            await self.service_manager.stop_all_services()
            self.logger.info("All services stopped.")
        self.logger.info("Orchestrator cleanup completed.")

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