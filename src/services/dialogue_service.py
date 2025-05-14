"""
Dialogue Service Module for Vtuber-AI
Handles the generation of responses using the LLM.
"""
import asyncio
from .base_service import BaseService
# Import necessary LLM utilities, prompt templates, etc.
# from utils.llm_utils import LLMUtils # Example
# from utils.prompt_template import LLMPromptTemplate # Example

class DialogueService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        self.llm_model = shared_resources.get("character_model")
        self.prompt_template = shared_resources.get("llm_prompt_template")
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

    async def run_worker(self):
        """Main logic for the Dialogue service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")

        if not self.llm_model or not self.prompt_template:
            if self.logger:
                self.logger.error("LLM model or prompt template not available in DialogueService. Stopping worker.")
            return

        if not self.speech_queue or not self.live_chat_queue or not self.llm_output_queue:
            if self.logger:
                self.logger.error("One or more required queues are missing in DialogueService. Stopping worker.")
            return

        try:
            while True:
                message = None
                try:
                    if self.speech_queue and not self.speech_queue.empty():
                        message = await self.speech_queue.get()
                        self.speech_queue.task_done()
                    elif self.live_chat_queue and not self.live_chat_queue.empty():
                        message = await self.live_chat_queue.get()
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

                parsed_speaker = self.user_name 
                raw_input_text = message
                if ": " in message:
                    try:
                        parsed_speaker, raw_input_text = message.split(": ", 1)
                    except ValueError:
                        if self.logger:
                            self.logger.warning(f"Could not parse speaker from message: {message}. Using raw message as input.")
                
                history_for_llm_content = "\n".join(list(self.naive_short_term_memory))
                content_for_template_hole = f"{self.prompt_template.instructions}\n\n{history_for_llm_content}\n{self.prompt_template.user_name}: {raw_input_text}\n{self.prompt_template.character_name}:"
                chatml_template = self.prompt_template.capybaraChatML

                if self.logger:
                    self.logger.debug(f"Content for LLM template hole: {content_for_template_hole[:200]}...")
                    self.logger.debug(f"ChatML template: {chatml_template[:200]}...")

                if not self.llm_output_queue.full():
                    output = await self.llm_model.dialogue_generator(content_for_template_hole, chatml_template, max_tokens=100)
                    
                    await self.llm_output_queue.put(output)
                    if self.logger:
                        self.logger.info(f"LLM generated response: {output[:100]}...")

                    self.naive_short_term_memory.append(message) 
                    self.naive_short_term_memory.append(f"{self.character_name}: {output}")
                    
                    if self.write_to_log_fn and self.conversation_log_file:
                        try:
                            msg_speaker, msg_text = message.split(": ", 1)
                            self.write_to_log_fn(self.conversation_log_file, msg_speaker, msg_text)
                        except ValueError:
                            if self.logger:
                                self.logger.warning(f"Could not parse speaker/text from incoming message for logging: {message}")
                            self.write_to_log_fn(self.conversation_log_file, "UnknownSpeaker", message)
                        
                        self.write_to_log_fn(self.conversation_log_file, self.character_name, output)
                else:
                    if self.logger:
                        self.logger.warning("LLM output queue is full, skipping generation.")

        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")