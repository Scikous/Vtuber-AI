import multiprocessing as mp
import heapq
from src.utils import logger as app_logger
from src.utils.app_utils import setup_project_root

def gpu_manager(request_queue, worker_events, max_slots):
    """
    Manages GPU access using a priority queue.
    
    Args:
        request_queue (mp.Queue): Queue for receiving acquire/release requests.
        max_slots (int): Maximum number of concurrent GPU slots.
    """
    setup_project_root()
    logger = app_logger.get_logger("GPUClientWorker")

    available_slots = max_slots
    waiting_queue = []  # Min-heap for (priority, worker_id) tuples

    while True:
        try:
            message = request_queue.get(timeout=0.1)
            
            if message["type"] == "acquire":
                priority = message["priority"]
                worker_id = message["worker_id"]
                if available_slots > 0:
                    available_slots -= 1
                    worker_events[worker_id].set()
                    logger.info(f"Granted GPU access to {worker_id}")
                else:
                    heapq.heappush(waiting_queue, (priority, worker_id))  # Queue the request
                    logger.info(f"Queued GPU access to {worker_id}")

            elif message["type"] == "release":
                available_slots += 1
                logger.info(f"{worker_id} released GPU")
                if waiting_queue:
                    next_priority, next_worker_id = heapq.heappop(waiting_queue)  # Get highest priority
                    worker_events[next_worker_id].set()  # Grant access
                    available_slots -= 1
                    logger.info(f"Granted GPU access to {next_worker_id}")

        except mp.queues.Empty:
            pass  # Wait for new messages