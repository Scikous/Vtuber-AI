#!/usr/bin/env python3
"""
LiveChat Worker Process for Vtuber-AI
Handles live chat processing in a dedicated process for optimal performance.
"""
import os
import sys
import logging
import time
import threading
from multiprocessing import Queue, Event

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import logger as app_logger

def livechat_process_worker(
    live_chat_queue: Queue,
    terminate_event: Event,
    immediate_livechat_fetch_event: Event,
    shared_config: dict
):
    """
    LiveChat worker process function.
    
    Args:
        live_chat_queue: Output queue to LLM
        terminate_event: Event to signal process termination
        immediate_livechat_fetch_event: Event to trigger immediate fetch
        shared_config: Configuration dictionary
    """
    # Setup logging for this process
    logger = app_logger.get_logger("LiveChat-Worker")
    logger.info("LiveChat worker process starting...")
    
    try:
        # Import LiveChat functionality
        from livechatAPI.livechat_controller import LiveChatController
        
        config = shared_config.get("config", {})
        
        # Initialize LiveChat controller
        logger.info("Initializing LiveChat controller...")
        
        # Get LiveChat configuration
        livechat_config = config.get("livechat", {})
        # fetch_twitch = livechat_config.get("fetch_twitch", False)
        # fetch_bilibili = livechat_config.get("fetch_bilibili", False)
        # fetch_youtube = livechat_config.get("fetch_youtube", False)
        
        # Initialize the controller based on configuration
        livechat_controller = None
        
        try:
            # This will depend on your specific LiveChat implementation
            livechat_controller = LiveChatController.create()
            logger.info("LiveChat controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LiveChat controller: {e}")
            logger.info("LiveChat worker will run in disabled mode")
        
        # Message processing settings
        fetch_interval = livechat_config.get("fetch_interval", 1.0)  # Default 1 second
        # max_messages_per_fetch = livechat_config.get("max_messages_per_fetch", 5)
        
        def fetch_and_process_messages():
            """Fetch and process live chat messages."""
            if not livechat_controller:
                return
            
            try:
                # Fetch new messages
                message, context_messages = livechat_controller.fetch_chat_message()
                
                if message and message.strip():
                    # Send to LLM queue
                    try:
                        live_chat_queue.put_nowait((message, context_messages))
                        logger.info(f"LiveChat message queued: {message[:50]}...")
                    except:
                        logger.warning(f"Failed to queue LiveChat message: {message[:30]}...")
            except Exception as e:
                logger.error(f"Error fetching live chat messages: {e}")
        
        def message_fetcher():
            """Continuous message fetching in separate thread."""
            last_fetch_time = 0
            
            while not terminate_event.is_set():
                try:
                    current_time = time.time()
                    
                    # Check if we should fetch immediately or wait for interval
                    should_fetch = (
                        immediate_livechat_fetch_event.is_set() or
                        (current_time - last_fetch_time) >= fetch_interval
                    )
                    
                    if should_fetch:
                        fetch_and_process_messages()
                        last_fetch_time = current_time
                        
                        # Clear immediate fetch event if it was set
                        if immediate_livechat_fetch_event.is_set():
                            immediate_livechat_fetch_event.clear()
                    
                    # Sleep for a short time to prevent busy waiting
                    time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Error in message fetcher: {e}")
                    time.sleep(1)  # Longer sleep on error
        
        def connection_monitor():
            """Monitor LiveChat connections and attempt reconnection if needed."""
            if not livechat_controller:
                return
            
            while not terminate_event.is_set():
                try:
                    # Check connection status using the new methods
                    if not livechat_controller.is_connected():
                        logger.warning("LiveChat connection lost, attempting reconnection...")
                        livechat_controller.reconnect()
                    
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in connection monitor: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start worker threads
        fetcher_thread = threading.Thread(target=message_fetcher, daemon=True, name="LiveChatFetcher")
        monitor_thread = threading.Thread(target=connection_monitor, daemon=True, name="LiveChatMonitor")
        
        fetcher_thread.start()
        monitor_thread.start()
        
        logger.info("LiveChat worker threads started")
        
        # Keep the main thread alive
        while not terminate_event.is_set():
            time.sleep(0.1)
        
        logger.info("Waiting for LiveChat threads to finish...")
        fetcher_thread.join(timeout=2)
        monitor_thread.join(timeout=2)
    
    except ImportError as e:
        logger.error(f"Failed to import LiveChat modules: {e}")
        logger.info("LiveChat worker running in disabled mode")
        
        # Run in disabled mode - just keep the process alive
        while not terminate_event.is_set():
            time.sleep(1)
    
    except Exception as e:
        logger.error(f"Unhandled exception in LiveChat worker: {e}", exc_info=True)
    finally:
        # Cleanup LiveChat controller
        if 'livechat_controller' in locals() and livechat_controller:
            try:
                if hasattr(livechat_controller, 'disconnect'):
                    livechat_controller.disconnect()
                logger.info("LiveChat controller disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting LiveChat controller: {e}")
        
        logger.info("LiveChat worker process shutting down...")