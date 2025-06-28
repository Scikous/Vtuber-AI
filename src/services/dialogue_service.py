"""Dialogue Service Module for Vtuber-AI
Handles the generation of responses using the LLM.
Manages its own LLM model loading and character configuration using a context manager.
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
        self.speech_queue = self.queues.get("speech_queue")
        self.live_chat_queue = self.queues.get("live_chat_queue")
        self.llm_output_queue = self.queues.get("llm_output_queue")

        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event())
        self.is_audio_streaming_event = shared_resources.get("is_audio_streaming_event", asyncio.Event())
        self.stt_is_listening_event = self.shared_resources.get("stt_is_listening_event", asyncio.Event())
        
        # Character configuration placeholders
        self.character_name = None
        self.user_name = None
        self.instructions = None
        
        # Initialize memory and logging
        self.llm_settings = self.config.get("llm_settings", {}) if self.config else {}
        self.naive_short_term_memory = deque(maxlen=self.llm_settings.get("short_term_memory_maxlen", 6))
        self.max_tokens = self.llm_settings.get("max_tokens", 512)
        self.wait_for_tts = self.llm_settings.get("wait_for_tts", 0.2)
        self.min_sentence_len = self.config.get("min_sentence_len", 8)
        self._setup_conversation_logging()
        
        if self.logger:
            self.logger.info("DialogueService initialized. LLM model and character will be loaded on service start.")
    
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
    
    def _load_character_config(self):
        """Load character personality information from JSON file."""
        if self.logger:
            self.logger.info("Loading character configuration...")
        
        character_info_json_path = self.llm_settings.get("character_info_json", "LLM_Wizard/characters/character.json")
        if not os.path.isabs(character_info_json_path):
            project_root = self.shared_resources.get("project_root")
            character_info_json_path = os.path.join(project_root, character_info_json_path)
        
        if not os.path.exists(character_info_json_path):
            if self.logger:
                self.logger.error(f"Character info JSON not found at: {character_info_json_path}")
            raise FileNotFoundError(f"Character info JSON not found at: {character_info_json_path}")

        self.instructions, self.user_name, self.character_name = load_character(character_info_json_path)
        if self.logger:
            self.logger.info(f"Character '{self.character_name}' configuration loaded.")


    async def run_worker(self):
        """Main logic for the Dialogue service worker."""
        if self.logger:
            self.logger.info(f"{self.__class__.__name__} worker starting.")

        if not all([self.speech_queue, self.live_chat_queue, self.llm_output_queue]):
            if self.logger:
                self.logger.error("One or more required queues are missing. Stopping worker.")
            return
            
        try:
            # 1. Load character personality config
            self._load_character_config()

            # 2. Create the LLM model configuration object
            model_config = LLMModelConfig(
                main_model=self.llm_settings.get("llm_model_path", "turboderp/Qwen2.5-VL-7B-Instruct-exl2"),
                tokenizer_model=self.llm_settings.get("tokenizer_model_path", "Qwen/Qwen2.5-VL-7B-Instruct"),
                revision=self.llm_settings.get("llm_model_revision", "8.0bpw"),
                character_name=self.character_name,
                instructions=self.instructions
            )
            
            if self.logger:
                self.logger.info("Attempting to load LLM model...")

            # 3. Use an async context manager to load the model and wrap the main loop
            # This loads the model once and ensures it's cleaned up automatically on exit.
            async with await VtuberExllamav2.load_model(config=model_config) as llm_character_model:
                if self.logger:
                    self.logger.info(f"Successfully loaded LLM model for character '{self.character_name}'. Ready for dialogue.")

                while True:
                    message = None
                    context = None
                    try:
                        # Prioritize speech queue over chat queue
                        if not self.speech_queue.empty():
                            message = await self.speech_queue.get()
                            self.speech_queue.task_done()
                        elif not self.live_chat_queue.empty():
                            message, context = await self.live_chat_queue.get()
                            self.live_chat_queue.task_done()
                        else:
                            await asyncio.sleep(0.1)
                            continue
                    except AttributeError as e: 
                        if self.logger:
                            self.logger.error(f"Queue error during get: {e}. Queues may not be initialized.")
                        await asyncio.sleep(1)
                        continue
                    
                    if not message: 
                        continue

                    if self.logger:
                        self.logger.info(f"Dialogue service received message: {message}")
                    
                    prompt = extract_name_message(message)
                    if context: 
                        prompt = prompt_wrapper(prompt, context=context)
                    
                    if self.logger:
                        self.logger.debug(f"Calling dialogue_generator for: {prompt[:100]}...")

                    # Use the model object from the context manager
                    async_job = await llm_character_model.dialogue_generator(
                        prompt, 
                        conversation_history=self.naive_short_term_memory,
                        images=None, 
                        max_tokens=self.max_tokens
                    )

                    full_string = ""
                    tts_buffer = ""
                    first_sentence_sent = False
                    
                    async for result in async_job:
                        if self.terminate_current_dialogue_event.is_set():
                            if self.logger:
                                self.logger.info("Interruption event set. Cancelling current dialogue generation.")
                            await llm_character_model.cancel_dialogue_generation()
                            # break # Exit the async for loop

                        chunk_text = result.get("text", "")
                        
                        if chunk_text:
                            full_string += chunk_text
                            tts_buffer += chunk_text
                            
                            if contains_sentence_terminator(chunk_text):
                                text_to_send_to_tts = tts_buffer.strip()
                                if text_to_send_to_tts and len(text_to_send_to_tts) >= self.min_sentence_len:
                                    await self.llm_output_queue.put(text_to_send_to_tts)
                                    if self.logger:
                                        self.logger.debug(f"Queued sentence for TTS: {text_to_send_to_tts[:50]}...")
                                    tts_buffer = ""


                                    if not first_sentence_sent:
                                        if self.logger:
                                            self.logger.info("Waiting for audio playback to start...")
                                        first_sentence_sent = True

                                        if not self.stt_is_listening_event.is_set():
                                            await llm_character_model.cancel_dialogue_generation()
                                            break
                                        await self.is_audio_streaming_event.wait()
                                        
                                        if self.logger:
                                            self.logger.info("Audio playback started. Resuming LLM generation.")
                                    
                                    elif self.llm_output_queue.full():
                                        if self.logger:
                                            self.logger.debug(f"TTS queue is full ({self.llm_output_queue.qsize()} items). Pausing generation...")
                                        await asyncio.sleep(self.wait_for_tts)
                        
                    # After the generation loop, send any remaining text in the buffer
                    if tts_buffer.strip():
                        await self.llm_output_queue.put(tts_buffer.strip())
                        if self.logger:
                            self.logger.debug(f"Queued remaining text for TTS: {tts_buffer.strip()[:50]}...")

                    if self.logger:
                        self.logger.info(f"LLM generated full response: {full_string[:100]}...")

                    self.naive_short_term_memory.append(message)
                    # self.naive_short_term_memory.append(f"{self.character_name}: {full_string}")
                    
                    if self.write_to_log_fn and self.conversation_log_file:
                        try:
                            msg_speaker, msg_text = message.split(": ", 1)
                            self.write_to_log_fn(self.conversation_log_file, (msg_speaker, msg_text))
                        except ValueError:
                            if self.logger:
                                self.logger.warning(f"Could not parse speaker/text for logging: {message}")
                            self.write_to_log_fn(self.conversation_log_file, ("UnknownSpeaker", message))
                        
                        self.write_to_log_fn(self.conversation_log_file, (self.character_name, full_string))
        except asyncio.CancelledError:
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker cancelled.")
        except FileNotFoundError as e:
            if self.logger:
                self.logger.error(f"Configuration file not found, stopping service: {e}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"A critical error occurred in {self.__class__.__name__} worker: {e}", exc_info=True)
        finally:
            # No explicit cleanup call is needed here. The 'async with' block ensures
            # the model's cleanup method is called when the block is exited,
            # either normally or through an exception/cancellation.
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} worker stopped. Model resources have been released.")