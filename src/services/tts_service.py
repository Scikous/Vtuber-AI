import asyncio
from TTS_Wizard import tts_client
from TTS_Wizard.tts_exp import XTTS_Service
from TTS_Wizard.tts_utils import prepare_tts_params_gpt_sovits, prepare_tts_params_xtts
from TTS_Wizard.realtimetts import RealTimeTTS
from .base_service import BaseService

class TTSService(BaseService):
    def __init__(self, shared_resources=None):
        super().__init__(shared_resources)
        self.queues = shared_resources.get("queues") if shared_resources else None
        self.llm_output_queue = self.queues.get("llm_output_queue") if self.queues else None
        self.logger = shared_resources.get("logger") if shared_resources else None
        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event()) if shared_resources else asyncio.Event()
        self.TTS_SERVICE = RealTimeTTS()#XTTS_Service("TTS_Wizard/dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav")
        self.prepare_tts_params = prepare_tts_params_xtts
    def synthesize_streaming(self, tts_params: dict):
        """
        Synthesize speech from text using the TTS module
        """
        return self.TTS_SERVICE.send_tts_request(**tts_params)

    async def _process_tts_item(self, tts_params: dict, semaphore: asyncio.Semaphore):
        """Helper function to process a single TTS request with semaphore control."""
        async with semaphore:
            if self.logger:
                self.logger.debug(f"TTS Service: Starting processing for: {str(tts_params.get('text', ''))[:50]}...")
            loop = asyncio.get_running_loop()
            audio_queue = self.shared_resources['queues']['audio_output_queue']
            
            def tts_request_handler():
                for audio_chunk in self.synthesize_streaming(tts_params):
                    if audio_chunk:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, audio_chunk)
                        if self.logger:
                            self.logger.debug(f"TTS Service: Put audio chunk of size {len(audio_chunk)} to audio_output_queue for text: {str(tts_params.get('text', ''))[:30]}...")
            
            try:
                await loop.run_in_executor(None, tts_request_handler)
                # async for audio_chunk in await self.synthesize_streaming(tts_params): # Removed await here as synthesize_streaming is an async generator
                #     if audio_chunk:
                #         await audio_queue.put(audio_chunk)
                #         if self.logger:
                #             self.logger.debug(f"TTS Service: Put audio chunk of size {len(audio_chunk)} to audio_output_queue for text: {str(tts_params.get('text', ''))[:30]}...")
                #     else:
                #         if self.logger:
                #             self.logger.debug(f"TTS Service: Received empty audio chunk for text: {str(tts_params.get('text', ''))[:30]}...")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during synthesize_streaming for {str(tts_params.get('text', ''))[:30]}: {e}", exc_info=True)
            finally:
                if self.llm_output_queue: # Check if queue exists before calling task_done
                    self.llm_output_queue.task_done()

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

                if self.terminate_current_dialogue_event.is_set() and (not self.llm_output_queue.empty() or active_tts_tasks):
                    while not self.llm_output_queue.empty():
                        try:
                            item = self.llm_output_queue.get_nowait()
                            self.llm_output_queue.task_done()
                            self.logger.debug(f"Discarded LLM output from queue due to termination.")
                        except asyncio.QueueEmpty:
                            break
                    if self.logger:
                        self.logger.info("TTS Service: Terminate current dialogue event set. Cancelling active TTS tasks...")
                    for task in active_tts_tasks:
                        if not task.done(): # Only cancel if not already done
                            task.cancel()
                    await asyncio.sleep(0.1) # Wait if queue is empty
                    # self.terminate_current_dialogue_event.clear()
                    continue
                    # asyncio.gather(*active_tts_tasks, return_exceptions=True) # Wait for tasks to finish cancellation

                if self.llm_output_queue and not self.llm_output_queue.empty():
                    # Only fetch new item if semaphore allows and we have less than max concurrent tasks active
                    # This check helps prevent overwhelming with too many scheduled tasks if semaphore is busy
                    if len(active_tts_tasks) < tts_concurrency * 2: # Allow some tasks to be queued up waiting for semaphore
                        try:
                            llm_message = await asyncio.wait_for(self.llm_output_queue.get(), timeout=0.1) 
                        except asyncio.TimeoutError:
                            await asyncio.sleep(0.05) # Short sleep if queue was empty during check
                            continue
                        except AttributeError: # Handle if queue is None
                            if self.logger:
                                self.logger.error("TTS Service: llm_output_queue is None.")
                            await asyncio.sleep(1)
                            continue

                        if self.logger:
                            self.logger.debug(f"TTS Service received message: {str(llm_message)[:100]}...")
                        

                        # tts_params = self.prepare_tts_params_gpt_sovits(llm_message,
                        #                                 text_lang=self.shared_resources.get("character_lang", "en"),
                        #                                 ref_audio_path=self.shared_resources.get("character_ref_audio_path", "../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav"),
                        #                                 prompt_text=self.shared_resources.get("character_prompt_text", ""),
                        #                                 prompt_lang=self.shared_resources.get("character_prompt_lang", "en"),
                        #                                 logger=self.logger
                        #                                 )
                        # tts_params = self.prepare_tts_params(llm_message)
                        # Create a new task to process this TTS item
                        # task = asyncio.create_task(self._process_tts_item(tts_params, semaphore))
                        # active_tts_tasks.append(task)
                        self.TTS_SERVICE.tts(llm_message)
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
