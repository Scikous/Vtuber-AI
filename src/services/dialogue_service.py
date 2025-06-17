"""Dialogue Service Module for Vtuber-AI
Handles the generation of responses using the LLM.
Now manages its own LLM model loading and character configuration.
"""
import asyncio
import os
from collections import deque
from LLM_Wizard.model_utils import contains_sentence_terminator, extract_name_message, prompt_wrapper, load_character
from LLM_Wizard.models import LLMModelConfig, VtuberExllamav2
from utils.file_operations import write_messages_csv
from utils.env_utils import get_env_var
from .base_service import BaseService

class DialogueService(BaseService):
    def __init__(self, shared_resources):
        super().__init__(shared_resources)
        
        # Queues for communication
        self.speech_queue = self.queues.get("speech_queue") # Input from STT
        self.live_chat_queue = self.queues.get("live_chat_queue") # Input from LiveChat
        self.llm_output_queue = self.queues.get("llm_output_queue") # Output to TTS/other consumers

        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event())
        self.is_audio_streaming_event = shared_resources.get("is_audio_streaming_event", asyncio.Event())
        
        # Service-managed resources
        self.llm_model = None
        self.character_name = None
        self.user_name = None
        self.instructions = None
        
        # Initialize memory and logging
        self.llm_settings = self.config.get("llm_settings", {}) if self.config else {}
        self.naive_short_term_memory = deque(maxlen=self.llm_settings.get("short_term_memory_maxlen", 6))
        self._setup_conversation_logging()
        
        if self.logger:
            self.logger.info("DialogueService initialized. LLM model will be loaded on service start.")
    
    def _setup_conversation_logging(self):
        """Setup conversation logging based on configuration."""
        self.conversation_log_file = get_env_var("CONVERSATION_LOG_FILE")
        if self.conversation_log_file and not os.path.isabs(self.conversation_log_file):
            project_root = self.shared_resources.get("project_root")
            self.conversation_log_file = os.path.join(project_root, self.conversation_log_file)
        
        self.write_to_log_fn = write_messages_csv if self.conversation_log_file else None
        
        if self.conversation_log_file and self.logger:
            self.logger.info(f"Conversation logging enabled to: {self.conversation_log_file}")
        elif self.logger:
            self.logger.info("Conversation logging is disabled.")
    
    async def _load_character_and_llm(self):
        """Load character information and LLM model."""
        if self.logger:
            self.logger.info("Loading character and LLM model...")
        
        # Load character information
        character_info_json_path = self.llm_settings.get("character_info_json", "LLM_Wizard/characters/character.json")
        if not os.path.isabs(character_info_json_path):
            project_root = self.shared_resources.get("project_root")
            character_info_json_path = os.path.join(project_root, character_info_json_path)
        
        if not os.path.exists(character_info_json_path):
            if self.logger:
                self.logger.error(f"Character info JSON not found at: {character_info_json_path}")
            raise FileNotFoundError(f"Character info JSON not found at: {character_info_json_path}")

        self.instructions, self.user_name, self.character_name = load_character(character_info_json_path)
        
        # Create LLM model configuration
        model_config = LLMModelConfig(
            main_model=self.llm_settings.get("llm_model_path", "turboderp/Qwen2.5-VL-7B-Instruct-exl2"),
            tokenizer_model=self.llm_settings.get("tokenizer_model_path", "Qwen/Qwen2.5-VL-7B-Instruct"),
            revision=self.llm_settings.get("llm_model_revision", "8.0bpw"),
            character_name=self.character_name,
            instructions=self.instructions
        )
        
        # Load the LLM model
        try:
            self.llm_model = VtuberExllamav2.load_model(config=model_config)
            if self.logger:
                self.logger.info(f"Successfully loaded LLM model for character '{self.character_name}'")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load LLM model: {e}")
            raise


    async def run_worker(self):
        """Main logic for the Dialogue service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker running.")
        
        # Load LLM model and character information
        try:
            await self._load_character_and_llm()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load LLM model in DialogueService: {e}")
            return

        if not self.speech_queue or not self.live_chat_queue or not self.llm_output_queue:
            if self.logger:
                self.logger.error("One or more required queues are missing in DialogueService. Stopping worker.")
            return
        output = ""
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
                first_chunk = True
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
                                # asyncio.create_task(self.llm_output_queue.put(text_to_send_to_tts))
                                await self.llm_output_queue.put(text_to_send_to_tts)
                                if self.logger:
                                    self.logger.debug(f"Put TTS params to llm_output_queue for sentence: {text_to_send_to_tts[:30]}...")
                                tts_buffer = "" # Reset buffer after sending
                                # if first_chunk:
                                #     first_chunk = False
                                #     await self.is_audio_streaming_event.wait()
                    else:
                        if self.logger:
                            self.logger.debug("Received empty chunk_text or chunk_text is None.")
                    
                # After the loop, if there's anything left in tts_buffer that wasn't sent
                # (e.g., the LLM finished generating mid-sentence)
                if tts_buffer.strip():
                    remaining_text_for_tts = tts_buffer.strip()
                    conditional_print("Sending remaining to TTS queue (end of generation): ", remaining_text_for_tts) # Keep for debugging
                    await self.llm_output_queue.put(remaining_text_for_tts)
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
            await self._cleanup_llm_model()
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped.")
    
    async def _cleanup_llm_model(self):
        """Clean up LLM model resources."""
        if self.llm_model:
            if self.logger:
                self.logger.info("Cleaning up LLM model resources...")
            try:
                await self.llm_model.cleanup()
                self.llm_model = None
                if self.logger:
                    self.logger.info("LLM model cleanup completed.")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during LLM model cleanup: {e}", exc_info=True)