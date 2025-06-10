from abc import ABC, abstractmethod
from model_utils import apply_chat_template, get_rand_token_len, character_reply_cleaner
import torch
import gc
import asyncio

class VtuberLLMBase(ABC):
    """Abstract base class for Vtuber LLM models."""
    def __init__(self, character_name, instructions):
        self.character_name = character_name
        self.instructions = instructions

    @abstractmethod
    def load_model(cls, *args, **kwargs):
        pass

    @abstractmethod
    async def dialogue_generator(self, prompt, **kwargs):
        pass

    @abstractmethod
    def cancel_dialogue_generation(self):
        """
        Requests the cancellation of the currently ongoing dialogue generation,
        if one exists and is cancellable.
        """
        pass

    @abstractmethod
    def cleanup(self):
        pass

    # Context manager methods -- used with "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        # Optionally, return False to propagate exceptions, True to suppress them
        return False

#current, lowest latency
class VtuberExllamav2(VtuberLLMBase):
    """
    Holds and handles all of the ExllamaV2 based tools
    """

    def __init__(self, generator, gen_settings, tokenizer, character_name, instructions):
        super().__init__(character_name, instructions)
        self.generator = generator
        self.gen_settings = gen_settings
        self.tokenizer = tokenizer
        self.current_async_job = None # Moved to VtuberLLMBase

    @classmethod
    def load_model(cls, main_model="turboderp/Qwen2.5-VL-7B-Instruct-exl2", tokenizer_model="Qwen/Qwen2.5-VL-7B-Instruct", revision="8.0bpw", character_name='assistant', instructions=""):
        """
        Loads an ExLlamaV2 compatible model

        main_model: str -- the actual model to use for generation
        tokenizer_model: str -- the tokenizer model to use for applying appropriate chat template
        revision: str -- the revision of the model to use
        character_name: str -- the name of the character the model is speaking as
        instructions: str -- the instructions for the model to follow

        Returns:
        
            generator (ExLlamaV2DynamicGenerator): to be used for text generation

            gen_settings (ExLlamaV2Sampler.Settings): default text generation settings -- reasonably unique responses
            
            tokenizer (ExLlamaV2Tokenizer): the initialized tokenizer -- given to the "model" in models.py
        """
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicGeneratorAsync, ExLlamaV2Sampler
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        
        
        #transformers tokenizer, not exllamav2's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model) # for applying chat template

        #load exllamav2 model
        try:
            config = ExLlamaV2Config(main_model)
        except:
            hf_model = snapshot_download(repo_id=main_model, revision=revision)
            config = ExLlamaV2Config(hf_model)

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len = 65536, lazy = True)
        model.load_autosplit(cache, progress = True)

        generator_async = ExLlamaV2DynamicGeneratorAsync(
            model = model,
            cache = cache,
            tokenizer = ExLlamaV2Tokenizer(config),
        )
        #default text generation settings, can be overridden
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature = 1.8, 
            top_p = 0.95,
            min_p=0.08,
            top_k=50,
            token_repetition_penalty = 1.05
        )

        return cls(generator_async, gen_settings, tokenizer, character_name, instructions)

    async def dialogue_generator(self, prompt, conversation_history=None, max_tokens=200):
        """
        Generates character's response to a given input (Message)

        For text length variety's sake, randomly selects token length to appear more natural
        """
        from exllamav2.generator import ExLlamaV2DynamicJobAsync
        prompt = apply_chat_template(instructions=self.instructions, prompt=prompt, conversation_history=conversation_history, tokenizer=self.tokenizer)

        self.current_async_job = ExLlamaV2DynamicJobAsync(
                        generator=self.generator,
                        encode_special_tokens=False,
                        decode_special_tokens=False,
                        completion_only=True,
                        input_ids=prompt,
                        max_new_tokens=max_tokens,
                        gen_settings=self.gen_settings,
                        add_bos = False, #if using apply_chat_template set to false -- only plain string should have True)
                        stop_conditions= [self.tokenizer.eos_token_id], #for stopping generation when a specific token is generated
                        #token_healing = False #True if output is weird, False if output is un-weird
                        #return_logits = False #for analyzing model's probability distribution before sapling -- generally don't touch
                        #return_probs = False #for understanding the model's confidence in its choices -- generally don't touch
                        #filters = None #list[list[ExLlamaV2Filter]] | list[ExLlamaV2Filter] forcing/guiding text generation
                        #identifier = None #object for tracking/associating metadata
                        #banned_strings = None #list[str] for banning specific words/phrases
                        #embeddings = None #list[ExLlamaV2MMEmbedding] can input images thathave been embedded into vectors

                    )
        return self.current_async_job    

    async def cancel_dialogue_generation(self):
        """
        Cancels the currently ongoing ExLlamaV2DynamicJobAsync.
        """
        if self.current_async_job:
            if hasattr(self.current_async_job, 'cancel') and callable(self.current_async_job.cancel):
                print("VtuberExllamav2: Cancelling current dialogue generation job.")
                try:
                    await self.current_async_job.cancel()
                except Exception as e:
                    print(f"VtuberExllamav2: Error trying to cancel async_job: {e}")
            else:
                print("VtuberExllamav2: current_async_job does not have a callable 'cancel' method.")
        else:
            print("VtuberExllamav2: No current dialogue generation job to cancel.")

    def cleanup(self):
        # Manual cleanup
        if self.current_async_job:
            print("Attempting to cancel ongoing async_job during cleanup...")
            self.cancel_dialogue_generation() # Calls the new cancel method
            self.current_async_job = None
        if hasattr(self, 'generator') and self.generator:
            del self.generator
            self.generator = None
        if hasattr(self, 'gen_settings') and self.gen_settings:
            del self.gen_settings
            self.gen_settings = None
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # If using CUDA
        gc.collect()
        print("Cleaned up and garbage collected ExllamaV2 Model resources!")

    # Context manager methods -- used with "with" statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        # Optionally, return False to propagate exceptions, True to suppress them
        return False



# #legacy model, high latency
# class VtuberLLM(VtuberLLMBase):
#     def __init__(self, model, tokenizer, character_name):
#         super().__init__(character_name)
#         self.model = model
#         self.tokenizer = tokenizer

#     @classmethod
#     def load_model(cls, base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name="", character_name='assistant'):
#         from peft import PeftModel, PeftConfig
#         from transformers import AutoModelForCausalLM, AutoTokenizer
        
#         model = AutoModelForCausalLM.from_pretrained(base_model_name,
#                                                         device_map="auto",
#                                                         trust_remote_code=False,
#                                                         revision="main")

#         if custom_model_name:
#             print(custom_model_name)
#             config = PeftConfig.from_pretrained(custom_model_name)
#             model = PeftModel.from_pretrained(
#                 model, custom_model_name, offload_folder="LLM/offload")

#         model.eval()
#         tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
#         return cls(model, tokenizer, character_name)

#     async def dialogue_generator(self, prompt):
#         """
#         Generates character's response to a given input (Message)
#         """
#         def is_incomplete_sentence(text):
#             return text.strip()[-1] not in {'.', '!', '?'}

#         max_attempts = 7
#         comment_tokenized = apply_chat_template(instructions="", prompt=prompt,tokenizer=self.tokenizer)
#         # print("HAAAAA:\n",comment_tokenized, self.tokenizer.bos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token, self.tokenizer.eos_token_id)
#         inputs = apply_chat_template(instructions="", prompt=prompt,tokenizer=self.tokenizer)

#         generated_text = ""
#         # print(len(comment_tokenized["input_ids"][0]))
#         for attempt in range(max_attempts):
#             max_new_tokens = get_rand_token_len(input_len=len(comment_tokenized["input_ids"][0]))
#             results = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
#                                           max_new_tokens=max_new_tokens,
#                                           top_p=0.8, top_k=50, temperature=1.1,
#                                           repetition_penalty=1.2, do_sample=True, num_return_sequences=10)
#             output = self.tokenizer.batch_decode(results, skip_special_tokens=True)[0]
#             # print(f"{'#' * 30}\n{output}\n{'#' * 30}")
#             output_clean = character_reply_cleaner(output, self.character_name).lower()
#             # print(f"{'#' * 30}\n{output_clean}\n{'#' * 30}")
#             generated_text = output_clean
#             if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
#                 break

#         # print("Text generation finished")
#         return generated_text