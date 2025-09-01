import gc
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from vllm.sampling_params import RequestOutputKind
import torch
from model_utils import apply_chat_template

from .base import BaseModelConfig, VtuberLLMAsyncBase, VtuberLLMBase

# --- Shared Logic ---
def _prepare_prompt(instance: Any,
                    prompt: str,
                    assistant_prompt: Optional[str]=None,
                    conversation_history: Optional[List[str]] = None,
                    images: Optional[List[Dict]] = None,
                    add_generation_prompt: bool = True,
                    continue_final_message: bool = False,
                    **kwargs) -> str:
    """Applies the chat template to build the final prompt string."""
    # This helper function is shared between sync and async classes
    if images and instance.config.is_vision_model:
        instance.logger.warning("Image data provided, but vLLM image handling is not yet implemented. Ignoring images.")
    
    return apply_chat_template(
        instructions=instance.instructions,
        prompt=prompt,
        assistant_prompt=assistant_prompt,
        conversation_history=conversation_history,
        tokenizer=instance.tokenizer,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )


def _create_sampling_params(generation_config: Optional[Dict[str, Any]] = None) -> SamplingParams:
    """
    Creates a vLLM SamplingParams object from a configuration dictionary.

    This helper function is responsible for parsing the generation configuration,
    specifically handling the dynamic creation of GuidedDecodingParams.

    Args:
        generation_config (Dict[str, Any], optional): A dictionary containing parameters
            for vLLM's SamplingParams, including a special 'guided_decoding' key.

    Returns:
        SamplingParams: A configured vLLM sampling parameters object.
    """
    config = generation_config.copy() if generation_config else {}

    # Handle guided decoding agnostically
    guided_decoding_config = config.pop("guided_decoding", None)
    guided_decoding_params = None

    if guided_decoding_config:
        # Expects a dict with a single key like 'json', 'regex', 'grammar'
        # e.g., {'json': '{"type": "object"}'}
        if isinstance(guided_decoding_config, dict) and len(guided_decoding_config) == 1:
            # The key is the type (json, regex), the value is the schema
            guided_decoding_params = GuidedDecodingParams(**guided_decoding_config)
        else:
            raise ValueError(
                "'guided_decoding' in generation_config must be a dictionary "
                "with a single key specifying the decoding type (e.g., 'json', 'regex')."
            )

    return SamplingParams(guided_decoding=guided_decoding_params, **config)


_DIALOGUE_GENERATOR_DOCSTRING = """
Generate text using the vLLM engine.

Args:
    prompt (str): The prompt to generate text from.
    assistant_prompt (str, optional): Optional prompt for the assistant to use as the base for its response.
    conversation_history (List, optional): List of previous messages in order [user_msg1, assistant_msg1, ...].
    images (List, optional): List of image dictionaries for vision models.
    add_generation_prompt (bool, optional): Whether to add the generation prompt. Defaults to True.
    continue_final_message (bool, optional): Whether to treat the prompt as a continuation of the assistant's message. Defaults to False.
    generation_config (Dict[str, Any], optional): A dictionary containing all sampling and generation parameters.
        This is used to create the `vllm.SamplingParams` object. For guided decoding, include a special
        'guided_decoding' key.

Example for `generation_config` with guided JSON:
generation_config = {{
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "guided_decoding": {{
        "json": '{{"type": "object", "properties": {{"name": {{"type": "string"}}}}}}'
    }}
}}

Returns:
{return_type}: {return_description}
"""

# --- Asynchronous vLLM Implementation ---
class VtuberVLLMAsync(VtuberLLMAsyncBase):
    """Asynchronous implementation of VtuberLLMBase using vLLM's AsyncLLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.current_request_id: Optional[str] = None

    @classmethod
    async def load_model(cls, config: BaseModelConfig) -> "VtuberVLLMAsync":
        from vllm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM AsyncLLM for model: {config.model_path_or_id}")
        engine_args = AsyncEngineArgs(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        trust_remote_code=True,
        **config.model_init_kwargs
        )
        instance.engine = AsyncLLM.from_engine_args(engine_args)
        instance.tokenizer = await instance.engine.get_tokenizer()
        return instance

    async def dialogue_generator(self,
                             prompt: str,
                             assistant_prompt: Optional[str] = None,
                             conversation_history: Optional[List[str]] = None,
                             images: Optional[List[Dict]] = None,
                             add_generation_prompt: bool = True,
                             continue_final_message: bool = False,
                             generation_config: Optional[Dict[str, Any]] = None
                             ) -> AsyncGenerator[str, None]:
        
        full_prompt = _prepare_prompt(self,
                                      prompt,
                                      assistant_prompt,
                                      conversation_history,
                                      images,
                                      add_generation_prompt,
                                      continue_final_message
                                                  )
        sampling_params = _create_sampling_params(generation_config)
        self.current_request_id = f"vtuber-llm-{uuid.uuid4().hex}"
        
        results_generator = self.engine.generate(full_prompt, sampling_params, self.current_request_id)
        
        async for request_output in results_generator:
            new_text = request_output.outputs[0].text
            yield new_text
        self.current_request_id = None

    # Set the docstring dynamically
    dialogue_generator.__doc__ = _DIALOGUE_GENERATOR_DOCSTRING.format(
        return_type="AsyncGenerator[str, None]",
        return_description="An asynchronous generator that yields text chunks as they are generated."
    )

    async def cancel_dialogue_generation(self):
        if self.current_request_id:
            await self.engine.abort(self.current_request_id)
            self.logger.info(f"Aborted vLLM request: {self.current_request_id}")
            self.current_request_id = None

    async def warmup(self):
        self.logger.info("Warming up the async vLLM engine...")
        try:
            async for _ in self.dialogue_generator(prompt="Hello"):
                pass
            self.logger.info("Async vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during async vLLM warmup: {e}")

    async def cleanup(self):
        if self.current_request_id:
            await self.cancel_dialogue_generation()
        self.engine.shutdown()
        del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up async vLLM resources.")

# --- Synchronous vLLM Implementation ---
class VtuberVLLM(VtuberLLMBase):
    """Synchronous implementation of VtuberLLMBase using vLLM's LLM."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.engine: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    @classmethod
    def load_model(cls, config: BaseModelConfig) -> "VtuberVLLM":
        from vllm import LLM
        instance = cls(config)
        instance.logger.info(f"Initializing vLLM LLM for model: {config.model_path_or_id}")
        instance.engine = LLM(
            model=config.model_path_or_id,
            **config.model_init_kwargs
        )
        instance.tokenizer = instance.engine.get_tokenizer()
        return instance

    def dialogue_generator(self,
                       prompt: str,
                       assistant_prompt: Optional[str] = None,
                       conversation_history: Optional[List[str]] = None,
                       images: Optional[List[Dict]] = None,
                       add_generation_prompt: bool = True,
                       continue_final_message: bool = False,
                       generation_config: Optional[Dict[str, Any]] = None
                       ) -> str:
        

        full_prompt = _prepare_prompt(self,
                                        prompt,
                                        assistant_prompt,
                                        conversation_history,
                                        images,
                                        add_generation_prompt,
                                        continue_final_message
                                        )
        
        sampling_params = _create_sampling_params(generation_config)
        outputs = self.engine.generate(full_prompt, sampling_params)
        return outputs[0].outputs[0].text

    # Set the docstring dynamically
    dialogue_generator.__doc__ = _DIALOGUE_GENERATOR_DOCSTRING.format(
        return_type="str",
        return_description="The complete generated text as a single string."
    )

    def cancel_dialogue_generation(self):
        self.logger.warning("Synchronous vLLM does not support cancellation of a running generator.")

    def warmup(self):
        self.logger.info("Warming up the sync vLLM engine...")
        try:
            # Consume the generator to ensure execution
            self.dialogue_generator(prompt="Hello")
            self.logger.info("Sync vLLM engine warmup complete.")
        except Exception as e:
            self.logger.error(f"An error occurred during sync vLLM warmup: {e}")

    def cleanup(self):
        del self.engine
        self.engine = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Cleaned up sync vLLM resources.")