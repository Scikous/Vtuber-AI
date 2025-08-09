from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

class TTSBase(ABC):
    """
    Abstract Base Class for Text-to-Speech systems.

    This class defines a standard interface for TTS operations, ensuring that
    different underlying engines can be used interchangeably by the application.
    """
    @abstractmethod
    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs: Any):
        """
        Initializes the TTS system.
        Subclasses will handle specific engine configurations and setup.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        print("Initializing BaseTTS.")
        pass

    @abstractmethod
    async def speak(self, text: str, **kwargs: Any):
        """
        Synthesizes the given text and plays it asynchronously.
        Implementations should handle text streaming, audio playback, and queuing.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stops any currently playing or queued audio immediately.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Cleans up resources, shuts down the engine, and performs any
        necessary finalization to prevent memory leaks or hanging processes.
        """
        pass