import multiprocessing as mp
import asyncio
from src.utils.env_utils import setup_project_root
from src.utils import logger as app_logger
from src.common import config as app_config
from src.utils.performance_utils import apply_system_optimizations, async_check_gpu_memory
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from Livechat_Wizard.data_models import UnifiedMessage # Assuming data_models.py is accessible from the project root

app_logger.setup_logging()
logger = app_logger.get_logger("ContextLLMWorker")

async def context_runner(shutdown_event, stt_stream_queue, llm_control_queue, livechat_output_queue,
                        user_has_stopped_speaking_event, gpu_ready_event, gpu_request_queue, worker_event, worker_id):
    config = app_config.load_config()
    context_settings = config.get("context_llm_settings", {})
    apply_system_optimizations(logger, use_cuda=True)

    livechat_settings = config.get("livechat_settings", {})
    moderation_prompt = livechat_settings.get("moderation_prompt", "Is the following message appropriate? Answer with 'safe' or 'unsafe'. Message: {content}")
    
    await asyncio.to_thread(gpu_ready_event.wait)
    logger.info("Context Worker: GPU worker ready, proceeding with model loading.")
    
    model_config = LLMModelConfig(
        main_model=context_settings.get("context_llm_model_path"),
        revision=context_settings.get("context_llm_model_revision"),
        tokenizer_model=context_settings.get("context_llm_tokenizer_model_path"),
        instructions=context_settings.get("context_llm_instructions"),
        is_vision_model=context_settings.get("context_llm_is_vision_model")
    )
    max_tokens = context_settings.get("context_llm_max_tokens")

    async with await VtuberExllamav2.load_model(config=model_config) as context_model:
        logger.info("✅ Context LLM model loaded.")
        
        initial_prompt = None
        tts_playback_approved = False

        while not shutdown_event.is_set():
            try:
                if initial_prompt is None and not livechat_output_queue.empty():
                    try:
                        message = livechat_output_queue.get_nowait()
                        #TTS will not play until event is set -- set it here, dirty way of doing it but it works
                        tts_playback_approved = True
                        user_has_stopped_speaking_event.set()
                        # Create a background task to process the message without blocking the main loop
                        asyncio.create_task(process_chat_message(
                            message, context_model, moderation_prompt, llm_control_queue,
                            gpu_request_queue, worker_event, worker_id
                        ))
                    except mp.queues.Empty:
                        pass # Race condition, another process/thread got it. Safe to ignore.

                if user_has_stopped_speaking_event.is_set() and not tts_playback_approved:
                    logger.info("Context Worker: User stopped speaking. Approving TTS playback.")
                    tts_playback_approved = True
                    try:
                        if initial_prompt and initial_prompt is not None:
                            transcript = stt_stream_queue.get(timeout=0.8)
                            logger.info("Context Worker: Sending finalized transcript")
                            llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": False, "continue_final_message": True})
                    except mp.queues.Empty:
                        pass
                    initial_prompt = None

                if not stt_stream_queue.empty():
                    tts_playback_approved = False
                    transcript = stt_stream_queue.get_nowait()
                    logger.info(f"Context Worker: Received sentence: '{transcript}'")
                    if initial_prompt is None:
                        initial_prompt = transcript
                        logger.info(f"Context Worker: Sending initial prompt: '{transcript}'")
                        try:
                            llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": True, "continue_final_message": False}, timeout=1)
                        except mp.queues.Full:
                            logger.warning("LLM control queue full, dropping message.")
                    else:
                        prompt = f"""
                        Original Phrase: "{initial_prompt}"
                        New Phrase: "{transcript}"
                        """
                        change_detected = False
                        try:
                            gpu_request_queue.put({"type": "acquire", "priority": 2, "worker_id":  worker_id})
                            worker_event.wait()
                            await async_check_gpu_memory(logger)
                            async for result in await context_model.dialogue_generator(prompt, max_tokens=max_tokens):
                                response_text = result.get("text", "").lower().strip()
                                if "yes" in response_text.lower():
                                    change_detected = True
                                    break
                        finally:
                            gpu_request_queue.put({"type": "release", "worker_id": worker_id})

                        if change_detected:
                            logger.warning(f"Context Worker: Meaning changed! New prompt: '{transcript}'")
                            llm_control_queue.put({"action": "interrupt"})
                            llm_control_queue.put({"action": "start", "prompt": transcript, "add_generation_prompt": True, "continue_final_message": False})
                            initial_prompt = transcript

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in context runner loop: {e}", exc_info=True)
                await asyncio.sleep(1)


async def process_chat_message(
    message: UnifiedMessage,
    context_model: VtuberExllamav2,
    moderation_prompt: str,
    llm_control_queue: mp.Queue,
    gpu_request_queue: mp.Queue,
    worker_event: mp.Event,
    worker_id: str
):
    """Handles the moderation and potential forwarding of a single chat message."""
    logger.info(f"Processing chat message from '{message.username}': '{message.content}'")
    
    # 1. Construct the moderation prompt
    prompt = moderation_prompt.format(username=message.username, content=message.content)
    
    verdict = "unsafe" # Default to unsafe
    try:
        # 2. Request GPU access with priority 2 (lower than STT)
        gpu_request_queue.put({"type": "acquire", "priority": 2, "worker_id": worker_id})
        worker_event.wait() # Wait until GPU access is granted
        
        logger.debug("GPU acquired for chat moderation.")
        await async_check_gpu_memory(logger)
        
        # 3. Run moderation check
        async for result in await context_model.dialogue_generator(prompt, max_tokens=10):
            response_text = result.get("text", "").lower().strip()
            if "safe" in response_text:
                verdict = "safe"
                break # We have our answer, no need to generate more
    
    except Exception as e:
        logger.error(f"Error during chat message moderation: {e}", exc_info=True)
    finally:
        # 4. ALWAYS release the GPU
        gpu_request_queue.put({"type": "release", "worker_id": worker_id})
        logger.debug("GPU released after chat moderation.")

    # 5. Act on the verdict
    if verdict == "safe":
        logger.info(f"Message from '{message.username}' deemed SAFE. Forwarding to main LLM.")
        # Format the prompt for the main LLM to respond to
        final_prompt = f"A user in the live chat named '{message.username}' said: '{message.content}'"
        try:
            llm_control_queue.put({
                "action": "start", 
                "prompt": final_prompt, 
                "add_generation_prompt": True, 
                "continue_final_message": False
            })
        except mp.queues.Full:
            logger.warning("LLM control queue was full. Dropping safe chat message.")
    else:
        logger.warning(f"Message from '{message.username}' deemed UNSAFE. Discarding.")


def context_llm_worker(shutdown_event: mp.Event, stt_stream_queue: mp.Queue, llm_control_queue: mp.Queue, livechat_output_queue: mp.Queue,
                      user_has_stopped_speaking_event: mp.Event, gpu_ready_event: mp.Event, gpu_request_queue: mp.Queue, worker_event: mp.Event):
    setup_project_root()
    worker_id = "Context_LLM"
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(context_runner(
            shutdown_event, stt_stream_queue, llm_control_queue, livechat_output_queue,
            user_has_stopped_speaking_event, gpu_ready_event, gpu_request_queue, worker_event, worker_id
        ))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Context LLM worker has shut down.")

