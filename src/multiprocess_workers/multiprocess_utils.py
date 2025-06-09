#!/usr/bin/env python3
"""
Multiprocessing Utilities for Vtuber-AI
Shared utilities and helper functions for multiprocess workers.
"""
import os
import sys
import time
import signal
import logging
import threading
from typing import Dict, Any, Optional, Callable
from multiprocessing import Queue, Event, Value, Process
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger

class ProcessManager:
    """
    Manages multiprocess workers with graceful shutdown and monitoring.
    """
    
    def __init__(self, name: str = "ProcessManager"):
        self.name = name
        self.logger = app_logger.get_logger(f"ProcessManager-{name}")
        self.processes: Dict[str, Process] = {}
        self.terminate_events: Dict[str, Event] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
    def register_process(self, process_name: str, target: Callable, args: tuple, kwargs: dict = None):
        """
        Register a process to be managed.
        
        Args:
            process_name: Unique name for the process
            target: Target function for the process
            args: Arguments for the target function
            kwargs: Keyword arguments for the target function
        """
        if kwargs is None:
            kwargs = {}
            
        # Create terminate event for this process
        terminate_event = Event()
        self.terminate_events[process_name] = terminate_event
        
        # Add terminate event to args if not already present
        if 'terminate_event' not in kwargs:
            kwargs['terminate_event'] = terminate_event
        
        # Create process
        process = Process(
            target=target,
            args=args,
            kwargs=kwargs,
            name=f"{self.name}-{process_name}",
            daemon=False
        )
        
        self.processes[process_name] = process
        self.logger.info(f"Registered process: {process_name}")
    
    def start_all_processes(self):
        """
        Start all registered processes.
        """
        self.logger.info(f"Starting {len(self.processes)} processes...")
        
        for name, process in self.processes.items():
            try:
                process.start()
                self.logger.info(f"Started process: {name} (PID: {process.pid})")
            except Exception as e:
                self.logger.error(f"Failed to start process {name}: {e}")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitor_processes,
            daemon=True,
            name=f"{self.name}-Monitor"
        )
        self.monitoring_thread.start()
        
        self.logger.info("All processes started and monitoring enabled")
    
    def _monitor_processes(self):
        """
        Monitor process health and restart if necessary.
        """
        while not self.shutdown_event.is_set():
            try:
                for name, process in self.processes.items():
                    if not process.is_alive() and not self.terminate_events[name].is_set():
                        self.logger.warning(f"Process {name} died unexpectedly, attempting restart...")
                        
                        # Try to restart the process
                        try:
                            # Create new process with same configuration
                            old_process = process
                            new_process = Process(
                                target=old_process._target,
                                args=old_process._args,
                                kwargs=old_process._kwargs,
                                name=old_process.name,
                                daemon=False
                            )
                            
                            # Replace in registry
                            self.processes[name] = new_process
                            new_process.start()
                            
                            self.logger.info(f"Restarted process: {name} (New PID: {new_process.pid})")
                        except Exception as e:
                            self.logger.error(f"Failed to restart process {name}: {e}")
                
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
                time.sleep(5)
    
    def shutdown_all_processes(self, timeout: float = 10.0):
        """
        Gracefully shutdown all processes.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        self.logger.info("Initiating graceful shutdown of all processes...")
        
        # Signal shutdown to monitoring thread
        self.shutdown_event.set()
        
        # Set terminate events for all processes
        for name, event in self.terminate_events.items():
            event.set()
            self.logger.debug(f"Sent terminate signal to process: {name}")
        
        # Wait for processes to terminate gracefully
        start_time = time.time()
        for name, process in self.processes.items():
            remaining_time = max(0, timeout - (time.time() - start_time))
            
            if process.is_alive():
                self.logger.info(f"Waiting for process {name} to terminate...")
                process.join(timeout=remaining_time)
                
                if process.is_alive():
                    self.logger.warning(f"Process {name} did not terminate gracefully, forcing termination")
                    process.terminate()
                    process.join(timeout=2)
                    
                    if process.is_alive():
                        self.logger.error(f"Process {name} could not be terminated, killing")
                        process.kill()
                        process.join(timeout=1)
                else:
                    self.logger.info(f"Process {name} terminated gracefully")
        
        # Wait for monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        self.logger.info("All processes shutdown complete")
    
    def get_process_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all managed processes.
        
        Returns:
            Dictionary with process status information
        """
        status = {}
        for name, process in self.processes.items():
            status[name] = {
                "alive": process.is_alive(),
                "pid": process.pid if process.is_alive() else None,
                "exitcode": process.exitcode,
                "terminate_requested": self.terminate_events[name].is_set()
            }
        return status

class QueueManager:
    """
    Manages multiprocessing queues with monitoring and cleanup.
    """
    
    def __init__(self, name: str = "QueueManager"):
        self.name = name
        self.logger = app_logger.get_logger(f"QueueManager-{name}")
        self.queues: Dict[str, Queue] = {}
        self.queue_stats: Dict[str, Dict[str, Any]] = {}
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
    
    def create_queue(self, queue_name: str, maxsize: int = 0) -> Queue:
        """
        Create and register a multiprocessing queue.
        
        Args:
            queue_name: Unique name for the queue
            maxsize: Maximum queue size (0 for unlimited)
            
        Returns:
            Created Queue object
        """
        queue = Queue(maxsize=maxsize)
        self.queues[queue_name] = queue
        self.queue_stats[queue_name] = {
            "created_time": time.time(),
            "maxsize": maxsize,
            "total_put": 0,
            "total_get": 0,
            "current_size": 0
        }
        
        self.logger.info(f"Created queue: {queue_name} (maxsize: {maxsize})")
        return queue
    
    def get_queue(self, queue_name: str) -> Optional[Queue]:
        """
        Get a registered queue by name.
        
        Args:
            queue_name: Name of the queue
            
        Returns:
            Queue object or None if not found
        """
        return self.queues.get(queue_name)
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start queue monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_queues,
            args=(interval,),
            daemon=True,
            name=f"{self.name}-QueueMonitor"
        )
        self.monitoring_thread.start()
        self.logger.info("Queue monitoring started")
    
    def _monitor_queues(self, interval: float):
        """
        Monitor queue statistics.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while not self.shutdown_event.is_set():
            try:
                for queue_name, queue in self.queues.items():
                    try:
                        # Update current size (approximate)
                        current_size = queue.qsize()
                        self.queue_stats[queue_name]["current_size"] = current_size
                        
                        # Log if queue is getting full
                        maxsize = self.queue_stats[queue_name]["maxsize"]
                        if maxsize > 0 and current_size > maxsize * 0.8:
                            self.logger.warning(
                                f"Queue {queue_name} is {current_size}/{maxsize} full ({current_size/maxsize*100:.1f}%)"
                            )
                    except Exception as e:
                        self.logger.debug(f"Error monitoring queue {queue_name}: {e}")
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in queue monitoring: {e}")
                time.sleep(interval)
    
    def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all queues.
        
        Returns:
            Dictionary with queue statistics
        """
        return self.queue_stats.copy()
    
    def cleanup_queues(self):
        """
        Clean up all queues.
        """
        self.logger.info("Cleaning up queues...")
        
        # Stop monitoring
        self.shutdown_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        # Drain all queues
        for queue_name, queue in self.queues.items():
            try:
                drained_count = 0
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        drained_count += 1
                    except:
                        break
                
                if drained_count > 0:
                    self.logger.info(f"Drained {drained_count} items from queue: {queue_name}")
            except Exception as e:
                self.logger.error(f"Error draining queue {queue_name}: {e}")
        
        self.logger.info("Queue cleanup complete")

@contextmanager
def process_timeout(seconds: float):
    """
    Context manager for process timeouts.
    
    Args:
        seconds: Timeout in seconds
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Process timed out after {seconds} seconds")
    
    # Set up signal handler (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout
        yield

def safe_queue_put(queue: Queue, item: Any, timeout: float = 1.0, logger: Optional[logging.Logger] = None) -> bool:
    """
    Safely put an item into a queue with timeout.
    
    Args:
        queue: Target queue
        item: Item to put
        timeout: Timeout in seconds
        logger: Optional logger for error reporting
        
    Returns:
        True if successful, False otherwise
    """
    try:
        queue.put(item, timeout=timeout)
        return True
    except Exception as e:
        if logger:
            logger.warning(f"Failed to put item in queue: {e}")
        return False

def safe_queue_get(queue: Queue, timeout: float = 1.0, logger: Optional[logging.Logger] = None) -> tuple[bool, Any]:
    """
    Safely get an item from a queue with timeout.
    
    Args:
        queue: Source queue
        timeout: Timeout in seconds
        logger: Optional logger for error reporting
        
    Returns:
        Tuple of (success, item)
    """
    try:
        item = queue.get(timeout=timeout)
        return True, item
    except Exception as e:
        if logger:
            logger.debug(f"Failed to get item from queue: {e}")
        return False, None

def drain_queue(queue: Queue, max_items: int = 1000, logger: Optional[logging.Logger] = None) -> int:
    """
    Drain items from a queue.
    
    Args:
        queue: Queue to drain
        max_items: Maximum number of items to drain
        logger: Optional logger for reporting
        
    Returns:
        Number of items drained
    """
    drained_count = 0
    
    try:
        while drained_count < max_items:
            try:
                queue.get_nowait()
                drained_count += 1
            except:
                break
    except Exception as e:
        if logger:
            logger.error(f"Error draining queue: {e}")
    
    if logger and drained_count > 0:
        logger.info(f"Drained {drained_count} items from queue")
    
    return drained_count

def setup_process_logging(process_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging for a multiprocess worker.
    
    Args:
        process_name: Name of the process
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = app_logger.get_logger(f"MP-{process_name}")
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    return logger

def create_shared_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a shared configuration dictionary for multiprocessing.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Shared configuration suitable for multiprocessing
    """
    # Create a copy that's safe for multiprocessing
    shared_config = {
        "config": config_dict.copy() if config_dict else {},
        "process_info": {
            "start_time": time.time(),
            "parent_pid": os.getpid()
        }
    }
    
    return shared_config

class PerformanceMonitor:
    """
    Monitor performance metrics for multiprocess workers.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = app_logger.get_logger(f"PerfMon-{name}")
        self.metrics = {
            "start_time": time.time(),
            "total_processed": 0,
            "total_errors": 0,
            "processing_times": [],
            "last_activity": time.time()
        }
        self.lock = threading.Lock()
    
    def record_processing_time(self, processing_time: float):
        """
        Record a processing time measurement.
        
        Args:
            processing_time: Time taken to process an item
        """
        with self.lock:
            self.metrics["processing_times"].append(processing_time)
            self.metrics["total_processed"] += 1
            self.metrics["last_activity"] = time.time()
            
            # Keep only recent measurements
            if len(self.metrics["processing_times"]) > 1000:
                self.metrics["processing_times"] = self.metrics["processing_times"][-500:]
    
    def record_error(self):
        """
        Record an error occurrence.
        """
        with self.lock:
            self.metrics["total_errors"] += 1
            self.metrics["last_activity"] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            processing_times = self.metrics["processing_times"]
            
            stats = {
                "name": self.name,
                "uptime": time.time() - self.metrics["start_time"],
                "total_processed": self.metrics["total_processed"],
                "total_errors": self.metrics["total_errors"],
                "error_rate": self.metrics["total_errors"] / max(1, self.metrics["total_processed"]),
                "last_activity": self.metrics["last_activity"],
                "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "throughput": self.metrics["total_processed"] / max(1, time.time() - self.metrics["start_time"])
            }
            
            return stats
    
    def log_stats(self):
        """
        Log current performance statistics.
        """
        stats = self.get_stats()
        self.logger.info(
            f"Performance Stats - Processed: {stats['total_processed']}, "
            f"Errors: {stats['total_errors']}, "
            f"Avg Time: {stats['avg_processing_time']:.3f}s, "
            f"Throughput: {stats['throughput']:.2f}/s"
        )