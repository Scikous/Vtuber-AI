"""
Dialogue Service Module for Vtuber-AI
Handles the generation of responses using the LLM.
"""
import asyncio
from LLM_Wizard.model_utils import contains_sentence_terminator, extract_name_message, prompt_wrapper
from .base_service import BaseService
# Import necessary LLM utilities, prompt templates, etc.

class DialogueService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.llm_model = shared_resources.get("character_model")
        self.naive_short_term_memory = shared_resources.get("naive_short_term_memory")
        self.character_name = shared_resources.get("character_name")
        self.user_name = shared_resources.get("user_name")
        self.speaker_name = shared_resources.get("speaker_name")
        self.conversation_log_file = shared_resources.get("conversation_log_file")
        self.write_to_log_fn = shared_resources.get("write_to_log_fn")

        # Queues for communication
        self.speech_queue = self.queues.get("speech_queue") # Input from STT
        self.live_chat_queue = self.queues.get("live_chat_queue") # Input from LiveChat
        self.llm_output_queue = self.queues.get("llm_output_queue") # Output to TTS/other consumers

        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event())


    async def run_worker(self):
        """Main logic for the Dialogue service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")

        if not self.llm_model:
            if self.logger:
                self.logger.error("LLM model not available in DialogueService. Stopping worker.")
            return

        if not self.speech_queue or not self.live_chat_queue or not self.llm_output_queue:
            if self.logger:
                self.logger.error("One or more required queues are missing in DialogueService. Stopping worker.")
            return

        try:
            while True:
                message = None
                context = None
                try:
                    if self.speech_queue and not self.speech_queue.empty():
                        message = await self.speech_queue.get()
                        self.speech_queue.task_done()
                    elif self.live_chat_queue and not self.live_chat_queue.empty():
                        message, context = await self.live_chat_queue.get()
                        self.live_chat_queue.task_done()
                    else:
                        await asyncio.sleep(0.1)
                        continue
                except AttributeError: 
                    if self.logger:
                        self.logger.error("DialogueService: Queues not properly initialized during get.")
                    await asyncio.sleep(1)
                    continue
                
                if not message: 
                    await asyncio.sleep(0.1)
                    continue

                if self.logger:
                    self.logger.info(f"Dialogue service received message: {message}")
                
                prompt = extract_name_message(message)#model_utils.prompt_wrapper(raw_input_text, history_for_llm_content)
                if context: prompt = prompt_wrapper(prompt, context=context)
                
                if self.logger:
                    self.logger.debug(f"Calling llm_model.dialogue_generator for: {prompt[:100]}...")

                async_job = await self.llm_model.dialogue_generator(prompt, conversation_history=self.naive_short_term_memory, max_tokens=100)
                if self.logger:
                    self.logger.debug(f"Got async_job: {type(async_job)}")

                full_string = ""
                tts_buffer = "" # Buffer for accumulating text for TTS

                async for result in async_job:
                    if self.terminate_current_dialogue_event.is_set():
                        if self.logger:
                            self.logger.info("Terminate current dialogue event set. Stopping DialogueService worker.")
                        await self.llm_model.cancel_dialogue_generation()
                        break
                    if self.logger:
                        self.logger.debug(f"Received result: {type(result)}")
                    chunk_text = result.get("text", "")
                    
                    if chunk_text and len(chunk_text) > 0:  # Ensure non-empty chunks are processed
                        full_string += chunk_text # Accumulate full response for memory/logging
                        tts_buffer += chunk_text
                        if contains_sentence_terminator(chunk_text):
                            text_to_send_to_tts = tts_buffer.strip()
                            if text_to_send_to_tts: # Ensure we don't send empty or whitespace-only strings
                                from utils.logger import conditional_print
                                 conditional_print("Sending to TTS queue: ", text_to_send_to_tts) # Keep for debugging
                                asyncio.create_task(self.llm_output_queue.put(text_to_send_to_tts))
                                if self.logger:
                                    self.logger.debug(f"Put TTS params to llm_output_queue for sentence: {text_to_send_to_tts[:30]}...")
                                tts_buffer = "" # Reset buffer after sending
                    else:
                        if self.logger:
                            self.logger.debug("Received empty chunk_text or chunk_text is None.")

                # After the loop, if there's anything left in tts_buffer that wasn't sent
                # (e.g., the LLM finished generating mid-sentence)
                if tts_buffer.strip():
                    remaining_text_for_tts = tts_buffer.strip()
                    conditional_print("Sending remaining to TTS queue (end of generation): ", remaining_text_for_tts) # Keep for debugging
                    asyncio.create_task(self.llm_output_queue.put(remaining_text_for_tts))
                    if self.logger:
                        self.logger.debug(f"Put remaining TTS params to llm_output_queue: {remaining_text_for_tts[:30]}...")
                    tts_buffer = "" # Clear the buffer

                output = full_string
                if self.logger:
                    self.logger.info(f"LLM generated response (first 100 chars): {output[:100]}...")

                self.naive_short_term_memory.append(message) 
                # self.naive_short_term_memory.append(f"{self.character_name}: {output}")
                
                if self.write_to_log_fn and self.conversation_log_file:
                    try:
                        msg_speaker, msg_text = message.split(": ", 1)
                        self.write_to_log_fn(self.conversation_log_file, (msg_speaker, msg_text))
                    except ValueError:
                        if self.logger:
                            self.logger.warning(f"Could not parse speaker/text from incoming message for logging: {message}")
                        self.write_to_log_fn(self.conversation_log_file, ("UnknownSpeaker", message))
                    
                    self.write_to_log_fn(self.conversation_log_file, (self.character_name, output))

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")