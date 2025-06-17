import asyncio
from .base_service import BaseService
from TTS_Wizard import tts_client
from TTS_Wizard.tts_exp import XTTS_Service
from TTS_Wizard.tts_utils import prepare_tts_params_gpt_sovits, prepare_tts_params_xtts, prepare_tts_params_rtts
from TTS_Wizard.realtimetts import RealTimeTTS

# A registry to map service names to their respective classes and param functions.
# This makes the service easily extensible.
TTS_SERVICE_REGISTRY = {
    "RealTimeTTS": {
        "class": RealTimeTTS,
        "params_fn": prepare_tts_params_rtts
    },
    "XTTS": {
        "class": XTTS_Service,
        "params_fn": prepare_tts_params_xtts
    },
    "GPT-SoVITS": {
        "class": tts_client, # Assuming tts_client is the class/module for this
        "params_fn": prepare_tts_params_gpt_sovits
    }
}

class TTSService(BaseService):
    def __init__(self, shared_resources=None):
        super().__init__(shared_resources)

        self.llm_output_queue = self.queues.get("llm_output_queue")
        self.audio_output_queue = self.queues.get("audio_output_queue")
        
        self.terminate_current_dialogue_event = self.shared_resources.get(
            "terminate_current_dialogue_event", asyncio.Event()
        )
        self.is_audio_streaming_event = self.shared_resources.get(
            "is_audio_streaming_event", asyncio.Event()
        )
        
        # --- Configuration-Driven TTS Initialization ---
        self.tts_settings = self.config.get("tts_settings", {}) if self.config else {}
        self.tts_concurrency = self.tts_settings.get("tts_concurrency", 2)
        tts_service_name = self.tts_settings.get("tts_service_name", "RealTimeTTS")
        service_config = TTS_SERVICE_REGISTRY.get(tts_service_name)
        if not service_config:
            raise ValueError(f"Unknown TTS service name: {tts_service_name}")
            
        # Instantiate the selected TTS service
        # NOTE: You might need to pass specific arguments to the service's __init__ method
        self.TTS_SERVICE = service_config["class"](is_audio_streaming_event=self.is_audio_streaming_event,terminate_current_dialogue_event=self.terminate_current_dialogue_event) 
        self.prepare_tts_params = service_config["params_fn"]
        
        
        if self.logger:
            self.logger.info(f"TTSService initialized with service: {tts_service_name}, concurrency: {self.tts_concurrency}")

    def synthesize_streaming(self, tts_params: dict):
        """Synthesize speech from text using the configured TTS module."""
        # This method remains generic and useful for queue-based services.
        return self.TTS_SERVICE.send_tts_request(**tts_params)

    async def _process_item_queued(self, tts_params: dict, semaphore: asyncio.Semaphore):
        """
        Processes a TTS request by generating audio chunks and putting them
        into the audio_output_queue. Used for services like XTTS.
        """
        async with semaphore:
            if self.logger:
                self.logger.debug(f"TTS (Queued): Starting processing for: {str(tts_params.get('text', ''))[:50]}...")
            
            loop = asyncio.get_running_loop()
            
            def tts_request_handler():
                for audio_chunk in self.synthesize_streaming(tts_params):
                    if audio_chunk:
                        loop.call_soon_threadsafe(self.audio_output_queue.put_nowait, audio_chunk)
            
            try:
                # Run the blocking, streaming TTS generation in a separate thread
                await loop.run_in_executor(None, tts_request_handler)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during queued synthesis for {str(tts_params.get('text', ''))[:30]}: {e}", exc_info=True)
            finally:
                semaphore.release()
                if self.llm_output_queue:
                    self.llm_output_queue.task_done()

    # async def _process_item_realtime(self, tts_params: dict):
    #     """
    #     Processes a TTS request using RealTimeTTS, which handles its own
    #     audio playback and does not use the audio_output_queue.
    #     """
    #     if self.logger:
    #         self.logger.debug(f"TTS (RealTime): Delegating playback for: {str(tts_params.get('text', ''))[:50]}...")
        
    #     try:
    #         # RealTimeTTS handles its own threading and playback. We can call it directly.
    #         # If its `tts` method is blocking, run_in_executor is safer.
    #         # Assuming `tts` is the correct method based on the original commented-out code.
    #         loop = asyncio.get_running_loop()
    #         await loop.run_in_executor(None, lambda: self.TTS_SERVICE.tts_request_async(**tts_params))
    #     except Exception as e:
    #         if self.logger:
    #             self.logger.error(f"Error during real-time synthesis for {str(tts_params.get('text', ''))[:30]}: {e}", exc_info=True)
    #     finally:
    #         if self.llm_output_queue:
    #             self.llm_output_queue.task_done()


    async def _process_item_realtime(self, tts_params: dict):
        """
        Processes a TTS request using RealTimeTTS by feeding text to its stream.
        """
        if self.logger:
            self.logger.debug(f"TTS (RealTime): Feeding text to stream: {str(tts_params.get('text', ''))[:50]}...")
        
        try:
            # The tts_request_async method is non-blocking. We can call it directly.
            # It will handle feeding the text and starting playback if necessary.
            self.TTS_SERVICE.tts_request_async(**tts_params)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during real-time synthesis for {str(tts_params.get('text', ''))[:30]}: {e}", exc_info=True)
        finally:
            # Crucially, mark the item from the input queue as done.
            if self.llm_output_queue:
                self.llm_output_queue.task_done()

        
    async def run_worker(self):
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running for {self.TTS_SERVICE.__class__.__name__}.")
        # Initialize semaphore for queue-based services -- RealTimeTTS does not utilize semaphore
        if not isinstance(self.TTS_SERVICE, RealTimeTTS):
            semaphore = asyncio.Semaphore(self.tts_concurrency)
        
        # Semaphore is used to control concurrency for queue-based services
        active_tts_tasks = []

        try:
            while True:
                active_tts_tasks = [task for task in active_tts_tasks if not task.done()]
                
                #terminate current dialogue when needed -- speech detected or terminate button smashed
                if self.terminate_current_dialogue_event.is_set():
                    # Termination logic remains the same
                    if not self.llm_output_queue.empty() or active_tts_tasks:
                        if self.logger:
                            self.logger.info("TTS Service: Terminating current dialogue. Clearing queue and cancelling tasks...")
                        while not self.llm_output_queue.empty():
                            try:
                                self.llm_output_queue.get_nowait()
                                self.llm_output_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        for task in active_tts_tasks:
                            task.cancel()
                        await asyncio.gather(*active_tts_tasks, return_exceptions=True)
                        active_tts_tasks.clear()
                        llm_message = None
                    await asyncio.sleep(0.1)
                    continue

                llm_message = await self.llm_output_queue.get()
                if self.logger:
                    self.logger.debug(f"TTS Service received message: {str(llm_message)[:100]}...")
                
                tts_params = self.prepare_tts_params(llm_message)
                
                task = None
                if isinstance(self.TTS_SERVICE, RealTimeTTS):
                    # RealTimeTTS manages its own concurrency, so we don't use our semaphore.
                    task = asyncio.create_task(self._process_item_realtime(tts_params))
                else:
                    # For queue-based services, wait for a free processing slot.
                    # This is the correct way to handle concurrency and maintain order.
                    # The loop will pause here if all slots are busy.
                    await semaphore.acquire()
                    
                    # Once a slot is acquired, create the task.
                    # The task is now responsible for releasing the semaphore.
                    task = asyncio.create_task(self._process_item_queued(tts_params, semaphore))
                
                if task:
                    active_tts_tasks.append(task)
                #if audio is NOT already playing DO NOT get new audio -- LLM will hog all the resources increacing latency dramatically
                # await self.is_audio_streaming_event.wait()

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled. Cleaning up...")
            for task in active_tts_tasks:
                task.cancel()
            await asyncio.gather(*active_tts_tasks, return_exceptions=True)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")