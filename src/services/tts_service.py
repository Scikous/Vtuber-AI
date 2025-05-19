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
        # Path to the TTS repo root (relative to this file or absolute)
        # self.tts_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TTS_Wizard/GPT_Test')) # Old path
        self.tts_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TTS_Wizard/GPT_SoVITS')) # New path
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

    async def synthesize_streaming(self, tts_params: dict):
        """
        Synthesize speech from text using the TTS repo's tts_exp.py interface (streaming).
        `tts_params` is a dictionary containing all necessary parameters for send_tts_request.
        """
        if hasattr(self.tts_module, 'send_tts_request'):
            # Pass logger if available
            if self.logger:
                tts_params['logger'] = self.logger
            # send_tts_request is now an async generator
            return self.tts_module.send_tts_request(**tts_params)
        else:
            if self.logger:
                self.logger.error('TTS module (tts_exp.py) does not have send_tts_request function.')
            raise RuntimeError('TTS module (tts_exp.py) does not have send_tts_request function.')

    # Optionally, add more methods to wrap other TTS repo functionality as needed

    async def _process_tts_item(self, tts_params: dict, semaphore: asyncio.Semaphore):
        """Helper function to process a single TTS request with semaphore control."""
        async with semaphore:
            if self.logger:
                self.logger.debug(f"TTS Service: Starting processing for: {str(tts_params.get('text', ''))[:50]}...")
            try:
                async for audio_chunk in await self.synthesize_streaming(tts_params): # Removed await here as synthesize_streaming is an async generator
                    if audio_chunk:
                        audio_queue = self.shared_resources['queues']['audio_output_queue']
                        await audio_queue.put(audio_chunk)
                        if self.logger:
                            self.logger.debug(f"TTS Service: Put audio chunk of size {len(audio_chunk)} to audio_output_queue for text: {str(tts_params.get('text', ''))[:30]}...")
                    else:
                        if self.logger:
                            self.logger.debug(f"TTS Service: Received empty audio chunk for text: {str(tts_params.get('text', ''))[:30]}...")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during synthesize_streaming for {str(tts_params.get('text', ''))[:30]}: {e}", exc_info=True)
            finally:
                if self.llm_output_queue: # Check if queue exists before calling task_done
                    self.llm_output_queue.task_done()
                if self.logger:
                    self.logger.debug(f"TTS Service: Finished processing for: {str(tts_params.get('text', ''))[:50]}...")

    async def run_worker(self):
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running with concurrent TTS processing.")
        
        tts_concurrency = self.shared_resources.get("tts_concurrency", 2) # Default to 2 concurrent tasks
        semaphore = asyncio.Semaphore(tts_concurrency)
        active_tts_tasks = []

        try:
            while True:
                # Clean up completed tasks
                active_tts_tasks = [task for task in active_tts_tasks if not task.done()]

                if self.llm_output_queue and not self.llm_output_queue.empty():
                    # Only fetch new item if semaphore allows and we have less than max concurrent tasks active
                    # This check helps prevent overwhelming with too many scheduled tasks if semaphore is busy
                    if len(active_tts_tasks) < tts_concurrency * 2: # Allow some tasks to be queued up waiting for semaphore
                        try:
                            tts_params_from_queue = await asyncio.wait_for(self.llm_output_queue.get(), timeout=0.1) 
                        except asyncio.TimeoutError:
                            await asyncio.sleep(0.05) # Short sleep if queue was empty during check
                            continue
                        except AttributeError: # Handle if queue is None
                            if self.logger:
                                self.logger.error("TTS Service: llm_output_queue is None.")
                            await asyncio.sleep(1)
                            continue

                        if self.logger:
                            self.logger.debug(f"TTS Service received params: {str(tts_params_from_queue)[:100]}...")
                        
                        if not all(k in tts_params_from_queue for k in ['text', 'text_lang', 'ref_audio_path', 'prompt_lang']):
                            if self.logger:
                                self.logger.error(f"TTS Service: Missing required parameters: {tts_params_from_queue}")
                            self.llm_output_queue.task_done()
                            continue
                        
                        # Create a new task to process this TTS item
                        task = asyncio.create_task(self._process_tts_item(tts_params_from_queue, semaphore))
                        active_tts_tasks.append(task)
                    else:
                        # Max tasks scheduled, wait for some to complete
                        await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.1) # Wait if queue is empty

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled. Waiting for active TTS tasks to complete...")
            for task in active_tts_tasks:
                task.cancel()
            await asyncio.gather(*active_tts_tasks, return_exceptions=True) # Wait for tasks to finish cancellation
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} active TTS tasks cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")