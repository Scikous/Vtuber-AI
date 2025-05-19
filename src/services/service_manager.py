"""
Service Manager Module for Vtuber-AI
Handles the lifecycle of all services.
"""
import asyncio

class ServiceManager:
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.logger = shared_resources.get("logger")
        self.services = []
        if self.logger:
            self.logger.info("ServiceManager initialized.")

    def register_service(self, service_instance):
        """Registers a service instance with the manager."""
        self.services.append(service_instance)
        if self.logger:
            self.logger.info(f"Registered service: {service_instance.__class__.__name__}")

    async def start_all_services(self):
        """Starts all registered services."""
        if self.logger:
            self.logger.info("Starting all registered services...")
        start_tasks = [service.start() for service in self.services]
        await asyncio.gather(*start_tasks)
        if self.logger:
            self.logger.info("All registered services started.")

    async def stop_all_services(self):
        """Stops all registered services."""
        if self.logger:
            self.logger.info("Stopping all registered services...")
        # Stop services in reverse order of registration, or as appropriate
        stop_tasks = [service.stop() for service in reversed(self.services)]
        await asyncio.gather(*stop_tasks, return_exceptions=True) # Allow all stop tasks to run even if one fails
        if self.logger:
            self.logger.info("All registered services stopped.")

    def get_service(self, service_class_name):
        """Retrieves a service instance by its class name."""
        for service in self.services:
            if service.__class__.__name__ == service_class_name:
                return service
        if self.logger:
            self.logger.warning(f"Service {service_class_name} not found.")
        return None