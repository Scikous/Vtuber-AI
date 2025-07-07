"""
Base Service Module for Vtuber-AI
Defines the abstract base class for all services.
"""
import asyncio
from abc import ABC, abstractmethod

class BaseService(ABC):
    def __init__(self, shared_resources):
        self.config = shared_resources.get("config", {}) if shared_resources else None
        self.logger = shared_resources.get("logger") if shared_resources else None
        self.queues = shared_resources.get("queues", {}) if shared_resources else None
        self.shared_resources = shared_resources # Store all shared resources
        self._task = None # To keep track of the running asyncio task

    @abstractmethod
    async def run_worker(self):
        """The main logic for the service worker. Must be implemented by subclasses."""
        pass

    async def start(self):
        """Starts the service worker as an asyncio task."""
        if self.logger:
            self.logger.info(f"Starting {self.__class__.__name__}...")
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run_worker())
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} started.")
        elif self.logger:
            self.logger.warning(f"{self.__class__.__name__} is already running or task exists.")

    async def stop(self):
        """Stops the service worker."""
        if self.logger:
            self.logger.info(f"Stopping {self.__class__.__name__}...")
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                if self.logger:
                    self.logger.info(f"{self.__class__.__name__} cancelled successfully.")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during {self.__class__.__name__} shutdown: {e}")
            finally:
                self._task = None
        elif self._task and self._task.done():
            if self.logger:
                self.logger.info(f"{self.__class__.__name__} was already stopped or completed.")
            self._task = None # Clear the task if it's done
        elif self.logger:
            self.logger.warning(f"{self.__class__.__name__} is not running or no task to stop.")

    def is_running(self):
        """Checks if the service worker task is currently running."""
        return self._task is not None and not self._task.done()