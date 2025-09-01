# llm_interface/base.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator

# --- Base Configuration ---
@dataclass
class BaseModelConfig:
    """Base configuration for any LLM model."""
    model_path_or_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    is_vision_model: bool = False
    max_seq_len: int = 4096
    character_name: str = 'assistant'
    instructions: str = ""
    model_init_kwargs: Dict[str, Any] = field(default_factory=dict)

# --- Synchronous Base Class ---
class VtuberLLMBase(ABC):
    """Abstract base class for synchronous Vtuber LLM models."""

    def __init__(self, config: BaseModelConfig, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.character_name = config.character_name
        self.instructions = config.instructions

    @classmethod
    @abstractmethod
    def load_model(cls, config: BaseModelConfig) -> "VtuberLLMBase":
        """Loads all necessary model resources and returns an instance of the class."""
        pass

    @abstractmethod
    def warmup(self):
        """Performs any necessary warmup operations."""
        pass

    @abstractmethod
    def dialogue_generator(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generates a response to a prompt, yielding tokens as they are generated."""
        pass

    @abstractmethod
    def cancel_dialogue_generation(self):
        """Requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleans up all resources held by the model."""
        pass

# --- Asynchronous Base Class ---
class VtuberLLMAsyncBase(VtuberLLMBase):
    """Abstract base class for asynchronous Vtuber LLM models."""

    @classmethod
    @abstractmethod
    async def load_model(cls, config: BaseModelConfig) -> "VtuberLLMAsyncBase":
        """Asynchronously loads all necessary model resources."""
        pass

    @abstractmethod
    async def warmup(self):
        """Asynchronously performs any necessary warmup operations."""
        pass

    @abstractmethod
    async def dialogue_generator(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Asynchronously generates a response to a prompt."""
        pass

    @abstractmethod
    async def cancel_dialogue_generation(self):
        """Asynchronously requests cancellation of the ongoing dialogue generation."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Asynchronously cleans up all resources held by the model."""
        pass

    async def __aenter__(self):
        """Async context manager for robust, async-aware cleanup."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensures cleanup is called upon exiting the context."""
        await self.cleanup()
        return False