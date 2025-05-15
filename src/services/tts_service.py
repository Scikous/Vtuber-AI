import os
import sys
import importlib.util
import asyncio
from typing import Optional
from .base_service import BaseService
from utils.app_utils import change_dir

class TTSService(BaseService):
    def __init__(self, shared_resources=None):
        super().__init__(shared_resources)
        # self.shared_resources = shared_resources
        # Path to the TTS repo root (relative to this file or absolute)
        self.tts_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TTS_Wizard/GPT_Test'))
        self.tts_exp_path = os.path.join(self.tts_repo_root, 'tts_exp.py')
        self.api_custom_path = os.path.join(self.tts_repo_root, 'api_custom.py')
        self._ensure_tts_dependencies()
        self.tts_module = self._import_tts_exp()
        
        # Initialize queue from shared resources
        self.queues = shared_resources.get("queues") if shared_resources else None
        self.llm_output_queue = self.queues.get("llm_output_queue") if self.queues else None
        self.logger = shared_resources.get("logger") if shared_resources else None

    def _ensure_tts_dependencies(self):
        # Add TTS repo root to sys.path if not already present
        if self.tts_repo_root not in sys.path:
            sys.path.insert(0, self.tts_repo_root)

    def _import_tts_exp(self):
        # Dynamically import tts_exp.py as a module
        spec = importlib.util.spec_from_file_location('tts_exp', self.tts_exp_path)
        tts_exp = importlib.util.module_from_spec(spec)
        sys.modules['tts_exp'] = tts_exp
        spec.loader.exec_module(tts_exp)
        return tts_exp

    def synthesize(self, text: str):
        """
        Synthesize speech from text using the TTS repo's tts_exp.py interface.
        """
        if hasattr(self.tts_module, 'send_tts_request'):
            return self.tts_module.send_tts_request(text)
        else:
            raise RuntimeError('TTS module does not have send_tts_request function.')

    # Optionally, add more methods to wrap other TTS repo functionality as needed

    async def run_worker(self):
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")
        try:
            while True:
                if self.llm_output_queue and not self.llm_output_queue.empty():
                    text = await self.llm_output_queue.get()
                        # Call the TTS repo
                    audio_bytes = await asyncio.to_thread(self.synthesize, text)
                    # print("audio_bytes", next(audio_bytes))
                    async for item in audio_bytes:
                        audio_queue = self.shared_resources['queues']['audio_output_queue']
                        await audio_queue.put(item)
                    self.llm_output_queue.task_done()
                else:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")