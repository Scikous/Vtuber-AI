# llm_interface/exllamav2.py

import asyncio
import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from model_utils import apply_chat_template, get_image

from .base import BaseModelConfig, VtuberLLMBase

# --- Configuration & Resource Objects ---
@dataclass
class Exllamav2ModelConfig(BaseModelConfig):
    """Configuration for loading an Exllamav2 model."""
    main_model: str
    tokenizer_model: str
    revision: str = "8.0bpw"
    max_seq_len: int = 65536
    is_vision_model: bool = True

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

# --- Implementation Class ---
class VtuberExllamav2(VtuberLLMBase):
    """
    Implementation of VtuberLLMBase using the ExllamaV2 library.
    """
    def __init__(self, config: Exllamav2ModelConfig, resources: Exllamav2Resources, logger: Optional[logging.Logger] = None):
        super().__init__(config.character_name, config.instructions, logger)
        self.config = config
        self.resources = resources

    @classmethod
    async def load_model(cls, config: Exllamav2ModelConfig) -> "VtuberExllamav2":
        """
        Loads an ExLlamaV2 model based on the provided configuration.
        """
        from exllamav2 import (ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config,
                               ExLlamaV2Tokenizer, ExLlamaV2VisionTower)
        from exllamav2.generator import (ExLlamaV2DynamicGeneratorAsync, ExLlamaV2Sampler)
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)

        try:
            exl2_config = ExLlamaV2Config(config.main_model)
        except FileNotFoundError:
            cls.logger.info(f"Local model not found. Downloading from Hugging Face Hub: {config.main_model}")
            hf_model_path = snapshot_download(repo_id=config.main_model, revision=config.revision)
            exl2_config = ExLlamaV2Config(hf_model_path)
        
        vision_model = None
        if config.is_vision_model:
            cls.logger.info("Loading vision tower...")
            vision_model = ExLlamaV2VisionTower(exl2_config)
            vision_model.load(progress=True)

        cls.logger.info("Loading main model...")
        model = ExLlamaV2(exl2_config)
        cache = ExLlamaV2Cache(model, max_seq_len=config.max_seq_len, lazy=True)
        model.load_autosplit(cache, progress=True)
        exll2_tokenizer = ExLlamaV2Tokenizer(exl2_config)
        
        generator = ExLlamaV2DynamicGeneratorAsync(model=model, cache=cache, tokenizer=exll2_tokenizer)
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature=1.8, top_p=0.95, min_p=0.08, top_k=50, token_repetition_penalty=1.05
        )

        loaded_resources = Exllamav2Resources(
            model=model, cache=cache, exll2tokenizer=exll2_tokenizer, 
            generator=generator, gen_settings=gen_settings, 
            tokenizer=hf_tokenizer, vision_model=vision_model
        )

        instance = cls(config, loaded_resources)
        return instance

    async def _prepare_prompt_and_embeddings(self, prompt: str, assistant_prompt: Optional[str], conversation_history: Optional[List[str]]=None, images: Optional[List[Dict]]=None,
                                            add_generation_prompt: bool = True, continue_final_message: bool = False):
        """Handles image embedding and chat template application."""
        image_embeddings = None
        placeholders = ""

        if self.resources.vision_model and images:
            loaded_images = [await get_image(**img_args) for img_args in images]
            
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
            assistant_prompt=assistant_prompt,
            conversation_history=conversation_history,
            tokenizer=self.resources.tokenizer,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        input_ids = self.resources.exll2tokenizer.encode(
            formatted_prompt, add_bos=True, encode_special_tokens=True, embeddings=image_embeddings
        )
        
        return input_ids, image_embeddings

    async def dialogue_generator(self, prompt: str, assistant_prompt: Optional[str]=None, conversation_history: Optional[List[str]] = None, images: Optional[List[Dict]] = None, max_tokens: int = 200, add_generation_prompt: bool = True, continue_final_message: bool = False):
        from exllamav2.generator import ExLlamaV2DynamicJobAsync

        input_ids, image_embeddings = await self._prepare_prompt_and_embeddings(
            prompt, assistant_prompt, conversation_history, images, add_generation_prompt, continue_final_message
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
        if self.current_async_job and hasattr(self.current_async_job, 'cancel'):
            self.logger.info("Cancelling current dialogue generation job.")
            try:
                await self.current_async_job.cancel()
            except Exception as e:
                self.logger.error(f"Error trying to cancel async_job: {e}")
        else:
            self.logger.warning("No cancellable dialogue generation job to cancel.")

    async def warmup(self):
        self.logger.info("Warming up the Exllamav2 LLM...")
        try:
            warmup_prompt = "Hello, world."
            warmup_job = await self.dialogue_generator(prompt=warmup_prompt, max_tokens=8)
            async for _ in warmup_job:
                pass
            self.current_async_job = None
            self.logger.info("Exllamav2 LLM warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during Exllamav2 LLM warmup: {e}")

    async def cleanup(self):
        if self.current_async_job:
            await self.cancel_dialogue_generation()
        
        if self.resources and self.resources.generator:
            await self.resources.generator.close()
        
        self.resources = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Cleaned up ExllamaV2 model resources.")