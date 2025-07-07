import asyncio
import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from model_utils import apply_chat_template, get_image

# Configure a logger for this module
log = logging.getLogger(__name__)


# --- Configuration & Resource Objects ---

@dataclass
class LLMModelConfig:
    """Configuration for loading an Exllamav2 model."""
    main_model: str = "turboderp/Qwen2.5-VL-7B-Instruct-exl2"
    tokenizer_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: str = "8.0bpw"
    is_vision_model: bool = True
    max_seq_len: int = 65536
    character_name: str = 'assistant'
    instructions: str = ""

@dataclass
class Exllamav2Resources:
    """A container for all loaded Exllamav2 resources."""
    model: Any
    cache: Any
    exll2tokenizer: Any
    generator: Any
    gen_settings: Any
    tokenizer: Any  # Transformers tokenizer
    vision_model: Optional[Any] = None


# --- Base Class ---


class VtuberLLMBase(ABC):
    """Abstract base class for Vtuber LLM models."""
    def __init__(self, character_name: str, instructions: str):
        self.character_name = character_name
        self.instructions = instructions
        self.current_async_job = None

    @classmethod
    @abstractmethod
    async def load_model(cls, *args, **kwargs):
        """Loads all necessary model resources and returns an instance of the class."""
        pass

    @abstractmethod
    async def warmup(self):
        """
        Loads the model weights and performs any necessary warmup operations.
        This method should be called before the model is ready to generate text.
        """
        pass

    

    @abstractmethod
    async def dialogue_generator(self, prompt: str, **kwargs):
        """Generates a response to a prompt."""
        pass

    @abstractmethod
    async def cancel_dialogue_generation(self):
        """Requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleans up all resources held by the model."""
        pass

    # Async context manager for robust, async-aware cleanup
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False # Propagate exceptions




#######
class VtuberExllamav2(VtuberLLMBase):
    """
    Implementation of VtuberLLMBase using the ExllamaV2 library.
    """
    def __init__(self, config: LLMModelConfig, resources: Exllamav2Resources):
        super().__init__(config.character_name, config.instructions)
        self.config = config
        self.resources = resources

    @classmethod
    async def load_model(cls, config: LLMModelConfig):
        """
        Loads an ExLlamaV2 model based on the provided configuration.
        """
        from exllamav2 import (ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config,
                               ExLlamaV2Tokenizer, ExLlamaV2VisionTower)
        from exllamav2.generator import (ExLlamaV2DynamicGeneratorAsync, ExLlamaV2Sampler)
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        # 1. Load Transformers Tokenizer (for chat template)
        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        # 2. Load Exllamav2 Model Config
        try:
            exl2_config = ExLlamaV2Config(config.main_model)
        except:
            log.info(f"Local model not found. Downloading from Hugging Face Hub: {config.main_model}")
            hf_model_path = snapshot_download(repo_id=config.main_model, revision=config.revision)
            exl2_config = ExLlamaV2Config(hf_model_path)
        
        # 3. Load Vision Model (if applicable)
        vision_model = None
        if config.is_vision_model:
            log.info("Loading vision tower...")
            vision_model = ExLlamaV2VisionTower(exl2_config)
            vision_model.load(progress=True)

        # 4. Load Core Model & Tokenizer
        log.info("Loading main model...")
        model = ExLlamaV2(exl2_config)
        cache = ExLlamaV2Cache(model, max_seq_len=config.max_seq_len, lazy=True)
        model.load_autosplit(cache, progress=True)
        exll2_tokenizer = ExLlamaV2Tokenizer(exl2_config)
        
        # 5. Initialize Generator and Sampler Settings
        generator = ExLlamaV2DynamicGeneratorAsync(model=model, cache=cache, tokenizer=exll2_tokenizer)
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature=1.8, top_p=0.95, min_p=0.08, top_k=50, token_repetition_penalty=1.05
        )

        # 6. Group resources and instantiate the class
        loaded_resources = Exllamav2Resources(
            model=model, cache=cache, exll2tokenizer=exll2_tokenizer, 
            generator=generator, gen_settings=gen_settings, 
            tokenizer=hf_tokenizer, vision_model=vision_model
        )

        instance = cls(config, loaded_resources)

        # 7. Perform warmup on the new instance
        await instance.warmup()

        return instance


    async def _prepare_prompt_and_embeddings(self, prompt: str, conversation_history: Optional[List[str]], images: Optional[List[Dict]]):
        """Handles image embedding and chat template application."""
        image_embeddings = None
        placeholders = ""

        if self.resources.vision_model and images:
            # Concurrently fetch and embed all images
            image_tasks = [get_image(**img_args) for img_args in images]
            loaded_images = await asyncio.gather(*image_tasks)
            
            image_embeddings = [
                self.resources.vision_model.get_image_embeddings(
                    model=self.resources.model,
                    tokenizer=self.resources.exll2tokenizer,
                    image=img
                )
                for img in loaded_images
            ]
            placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

        full_prompt = placeholders + prompt
        formatted_prompt = apply_chat_template(
            instructions=self.instructions,
            prompt=full_prompt,
            conversation_history=conversation_history,
            tokenizer=self.resources.tokenizer,
            tokenize=False
        )
        
        input_ids = self.resources.exll2tokenizer.encode(
            formatted_prompt, add_bos=True, encode_special_tokens=True, embeddings=image_embeddings
        )
        
        return input_ids, image_embeddings

    async def dialogue_generator(self, prompt: str, conversation_history: Optional[List[str]] = None, images: Optional[List[Dict]] = None, max_tokens: int = 200):
        """
        Generates character's response asynchronously.
        """
        from exllamav2.generator import ExLlamaV2DynamicJobAsync

        input_ids, image_embeddings = await self._prepare_prompt_and_embeddings(
            prompt, conversation_history, images
        )
        
        self.current_async_job = ExLlamaV2DynamicJobAsync(
            generator=self.resources.generator,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            gen_settings=self.resources.gen_settings,
            completion_only=True,
            add_bos=False,
            stop_conditions=[self.resources.tokenizer.eos_token_id],
            embeddings=image_embeddings
        )
        return self.current_async_job

    async def cancel_dialogue_generation(self):
        """Cancels the currently ongoing ExLlamaV2DynamicJobAsync."""
        if self.current_async_job and hasattr(self.current_async_job, 'cancel'):
            log.info("Cancelling current dialogue generation job.")
            try:
                await self.current_async_job.cancel()
            except Exception as e:
                log.error(f"Error trying to cancel async_job: {e}")
        else:
            log.warning("No cancellable dialogue generation job to cancel.")

    async def warmup(self):
        """
        Warms up the model by running a short, dummy generation.
        This pre-compiles CUDA kernels and initializes memory allocations,
        reducing latency on the first real generation request.
        """
        log.info("Warming up the LLM... (This may take a moment)")
        try:
            # 1. Define a simple, short prompt. No need for history or images.
            warmup_prompt = "Hello, world."
            images = [
                # {"file": "media/test_image_1.jpg"},
                # {"file": "media/test_image_2.jpg"},
                # {"url": "https://media.istockphoto.com/id/1212540739/photo/mom-cat-with-kitten.jpg?s=612x612&w=0&k=20&c=RwoWm5-6iY0np7FuKWn8FTSieWxIoO917FF47LfcBKE="},
                {"url": "https://i.dailymail.co.uk/1s/2023/07/10/21/73050285-12283411-Which_way_should_I_go_One_lady_from_the_US_shared_this_incredibl-a-4_1689019614007.jpg"},
                {"url": "https://images.fineartamerica.com/images-medium-large-5/metal-household-objects-trevor-clifford-photography.jpg"}
            ]
            dummy_memory =[
                "Bob: The venerable oak tree, standing as an immutable sentinel through countless seasons, its gnarled branches reaching skyward like ancient, petrified arms, silently bore witness to the fleeting dramas of human endeavor unfolding beneath its rustling canopy, embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization.",
                "As the crimson sun dipped below the jagged horizon, casting long, ethereal shadows across the ancient, crumbling ruins, a lone figure, cloaked in worn, travel-stained fabric, paused to contemplate the vast, silent expanse of the desolate wasteland stretching endlessly before them, a chilling premonition of trials yet to come slowly solidifying in the depths of their weary soul.",
                "Bob: Scientists, meticulously analyzing the arcane data collected from the deepest recesses of the oceanic trenches, discovered astonishing, bioluminescent organisms exhibiting previously unknown adaptive mechanisms, providing tantalizing insights into the astonishing resilience of life in environments once deemed utterly inhospitable to any form of complex existence.",
                "Despite the overwhelming complexities",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "multi-year development cycle, the dedicated team of engineers, fueled by an unyielding passion for innovation and an unwavering commitment to their ambitious vision, ultimately managed to revolutionize the nascent field of quantum computing with their groundbreaking, paradigm-shifting invention.",
                "Bob: The venerable oak tree",
                "standing as an immutable sentinel through countless seasons",
                "Bob: its gnarled branches reaching skyward like ancient",
                "petrified arms",
                "Bob: silently bore witness to the fleeting dramas of human endeavor unfolding beneath its rustling canopy",
                "embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization.",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "multi-year development cycle, the dedicated team of engineers, fueled by an unyielding passion for innovation and an unwavering commitment to their ambitious vision, ultimately managed to revolutionize the nascent field of quantum computing with their groundbreaking, paradigm-shifting invention.",
                "Bob: Scientists, meticulously analyzing the arcane data collected from the deepest recesses of the oceanic trenches, discovered astonishing, bioluminescent organisms exhibiting previously unknown adaptive mechanisms, providing tantalizing insights into the astonishing resilience of life in environments once deemed utterly inhospitable to any form of complex existence.",
                "Despite the overwhelming complexities",
                "Bob: and numerous unforeseen obstacles encountered during the arduous",
                "embodying a timeless wisdom far exceeding the ephemeral lifespan of any transient civilization."
            ]


            warmup_job = await self.dialogue_generator(prompt=warmup_prompt, conversation_history=dummy_memory, images=images, max_tokens=512)

            # 4. Await the job's result to ensure it runs to completion.
            # We don't care about the output, just that the computation happens.
            async for _ in warmup_job:
                pass
                

            # 5. Clear the job reference to avoid confusion with real jobs.
            # The dialogue_generator method will set self.current_async_job later.
            self.current_async_job = None
            
            log.info("LLM warmup complete. Model is ready.")

        except Exception as e:
            # Log an error but don't prevent the application from starting.
            log.error(f"An error occurred during LLM warmup: {e}")

    async def cleanup(self):
        """Asynchronously cleans up model resources."""
        if self.current_async_job:
            log.info("Attempting to cancel ongoing async_job during cleanup...")
            await self.cancel_dialogue_generation()
            self.current_async_job = None

        # Release resources
        self.resources = None
        
        # Clear CUDA cache and run garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("Cleaned up ExllamaV2 model resources.")




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