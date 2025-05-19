"""
Dialogue Service Module for Vtuber-AI
Handles the generation of responses using the LLM.
"""
import asyncio
from .base_service import BaseService
from LLM_Wizard.model_utils import LLMUtils
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
                content_for_template_hole = LLMUtils.prompt_wrapper(raw_input_text, history_for_llm_content)
                # if self.logger:
                #     self.logger.debug(f"Content for LLM template hole: {content_for_template_hole[:200]}...")
                if not self.llm_output_queue.full():
                    if self.logger:
                        self.logger.debug(f"Calling llm_model.dialogue_generator for: {content_for_template_hole[:100]}...")
                    async_job = await self.llm_model.dialogue_generator(content_for_template_hole, max_tokens=100)
                    if self.logger:
                        self.logger.debug(f"Got async_job: {type(async_job)}")
                    full_string = ""
                    tts_buffer = "" # Buffer for accumulating text for TTS
                    iteration_count = 0
                    sentence_terminators = [',','.', '!', '?']

                    async for result in async_job:
                        iteration_count += 1
                        if self.logger:
                            self.logger.debug(f"Async_job iteration {iteration_count}. Received result: {type(result)}")
                        chunk_text = result.get("text", "")
                        
                        if chunk_text and len(chunk_text) > 0:  # Ensure non-empty chunks are processed
                            full_string += chunk_text # Accumulate full response for memory/logging
                            tts_buffer += chunk_text

                            # Process buffer for complete sentences
                            while True:
                                split_point = -1
                                # Find the last occurrence of any sentence terminator
                                for terminator in sentence_terminators:
                                    current_split_point = tts_buffer.rfind(terminator)
                                    if current_split_point > split_point:
                                        split_point = current_split_point
                                
                                if split_point != -1:
                                    sentence_to_send = tts_buffer[:split_point+1].strip()
                                    tts_buffer = tts_buffer[split_point+1:]
                                    
                                    if sentence_to_send: # Ensure we are sending non-empty sentence
                                        print("Sending to TTS queue: ", sentence_to_send)
                                        # Ensure ref_audio_path is valid
                                        ref_audio_path = self.shared_resources.get("character_ref_audio_path", "../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav")
                                        if not ref_audio_path or not isinstance(ref_audio_path, str) or not ref_audio_path.strip():
                                            if self.logger:
                                                self.logger.warning(f"Invalid or missing ref_audio_path ('{ref_audio_path}'), falling back to default.")
                                            ref_audio_path = "../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav" # Known good default
                                        
                                        tts_params = {
                                            "text": sentence_to_send,
                                            "text_lang": self.shared_resources.get("character_lang", "en"),
                                            "ref_audio_path": ref_audio_path,#self.shared_resources.get("character_ref_audio_path", "../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav"),
                                            "prompt_text": self.shared_resources.get("character_prompt_text", ""),
                                            "prompt_lang": self.shared_resources.get("character_prompt_lang", "en"),
                                            "streaming_mode": False, # Keep true for chunk-by-chunk audio generation if supported by TTS
                                            "media_type": "wav",
                                        }
                                        
#                                     tts_params = {
#                                         "text": text_to_send_chunk, # Send the whole chunk
#                                         "text_lang": self.shared_resources.get("character_lang", "en"),
#                                         "ref_audio_path": ref_audio_path,
#                                         "prompt_text": self.shared_resources.get("character_prompt_text", ""),
#                                         "prompt_lang": self.shared_resources.get("character_prompt_lang", "en"),
#                                         "streaming_mode": False, # Static noise issue fix
#                                         "media_type": "wav",
#                                     }
                                        asyncio.create_task(self.llm_output_queue.put(tts_params))
                                        if self.logger:
                                            self.logger.debug(f"Put TTS params to llm_output_queue for sentence: {sentence_to_send[:30]}...")
                                else:
                                    break # No more complete sentences in buffer
                        else:
                            if self.logger:
                                self.logger.debug("Received empty chunk_text.")

                    # After the loop, send any remaining text in the buffer
                    if tts_buffer.strip():
                        tts_params = {
                            "text": tts_buffer.strip(),
                            "text_lang": self.shared_resources.get("character_lang", "en"),
                            "ref_audio_path": self.shared_resources.get("character_ref_audio_path", "../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav"),
                            "prompt_text": self.shared_resources.get("character_prompt_text", ""),
                            "prompt_lang": self.shared_resources.get("character_prompt_lang", "en"),
                            "streaming_mode": False, # Static noise fix - disable streaming
                            "media_type": "wav",
                        }
                        asyncio.create_task(self.llm_output_queue.put(tts_params))
                        if self.logger:
                            self.logger.debug(f"Put remaining TTS params to llm_output_queue: {tts_buffer.strip()[:30]}...")

                    output = full_string
                    # print("Wjssjsj"*1000) # Removed debug print
                    if self.logger:
                        self.logger.info(f"LLM generated response (first 100 chars): {output[:100]}...")
                        self.logger.debug(f"Total iterations for async_job: {iteration_count}")

                    self.naive_short_term_memory.append(message) 
                    self.naive_short_term_memory.append(f"{self.character_name}: {output}")
                    
                    if self.write_to_log_fn and self.conversation_log_file:
                        try:
                            msg_speaker, msg_text = message.split(": ", 1)
                            await self.write_to_log_fn(self.conversation_log_file, (msg_speaker, msg_text))
                        except ValueError:
                            if self.logger:
                                self.logger.warning(f"Could not parse speaker/text from incoming message for logging: {message}")
                            await self.write_to_log_fn(self.conversation_log_file, ("UnknownSpeaker", message))
                        
                        await self.write_to_log_fn(self.conversation_log_file, (self.character_name, output))
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
