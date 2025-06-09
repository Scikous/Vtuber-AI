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
            livechat_controller = LiveChatController()
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
            """Monitor LiveChat connection health."""
            if not livechat_controller:
                return
            
            while not terminate_event.is_set():
                try:
                    # Check connection health
                    if hasattr(livechat_controller, 'is_connected'):
                        if not livechat_controller.is_connected():
                            logger.warning("LiveChat connection lost, attempting to reconnect...")
                            try:
                                livechat_controller.reconnect()
                                logger.info("LiveChat reconnected successfully")
                            except Exception as e:
                                logger.error(f"Failed to reconnect to LiveChat: {e}")
                    
                    time.sleep(5)  # Check every 5 seconds
                
                except Exception as e:
                    logger.error(f"Error in connection monitor: {e}")
                    time.sleep(10)  # Longer sleep on error
        
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

def create_optimized_livechat_worker(
    live_chat_queue: Queue,
    terminate_event: Event,
    immediate_livechat_fetch_event: Event,
    shared_config: dict
):
    """
    Optimized LiveChat worker with advanced message filtering and batching.
    """
    logger = app_logger.get_logger("LiveChat-Worker-Optimized")
    logger.info("Optimized LiveChat worker starting...")
    
    try:
        from livechatAPI.livechat_controller import LiveChatController
        import re
        from collections import deque
        
        config = shared_config.get("config", {})
        livechat_config = config.get("livechat", {})
        
        # Advanced filtering settings
        min_message_length = livechat_config.get("min_message_length", 3)
        max_message_length = livechat_config.get("max_message_length", 200)
        spam_filter_enabled = livechat_config.get("spam_filter_enabled", True)
        duplicate_filter_window = livechat_config.get("duplicate_filter_window", 30)  # seconds
        
        # Message history for duplicate detection
        recent_messages = deque(maxlen=100)
        
        # Spam detection patterns
        spam_patterns = [
            r'^[!@#$%^&*()_+\-=\[\]{};":,.<>?/~`]+$',  # Only special characters
            r'^(.)\1{10,}$',  # Repeated characters
            r'https?://\S+',  # URLs (optional filtering)
        ]
        
        def is_spam_message(message):
            """Check if message is spam."""
            if not spam_filter_enabled:
                return False
            
            # Length checks
            if len(message) < min_message_length or len(message) > max_message_length:
                return True
            
            # Pattern checks
            for pattern in spam_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return True
            
            return False
        
        def is_duplicate_message(message, author):
            """Check if message is a recent duplicate."""
            current_time = time.time()
            message_key = f"{author}:{message.lower().strip()}"
            
            # Check recent messages
            for msg_time, msg_key in recent_messages:
                if current_time - msg_time < duplicate_filter_window:
                    if msg_key == message_key:
                        return True
            
            # Add to recent messages
            recent_messages.append((current_time, message_key))
            return False
        
        def process_message_batch(messages):
            """Process a batch of messages with advanced filtering."""
            processed_messages = []
            
            for message_data in messages:
                message = message_data.get("message", "")
                author = message_data.get("author", "Viewer")
                context = message_data.get("context", [])
                
                # Apply filters
                if is_spam_message(message):
                    logger.debug(f"Filtered spam message: {message[:30]}...")
                    continue
                
                if is_duplicate_message(message, author):
                    logger.debug(f"Filtered duplicate message: {message[:30]}...")
                    continue
                
                # Message passed filters
                processed_messages.append({
                    "message": f"{author}: {message.strip()}",
                    "context": context,
                    "priority": calculate_message_priority(message, author)
                })
            
            # Sort by priority (higher priority first)
            processed_messages.sort(key=lambda x: x["priority"], reverse=True)
            
            return processed_messages
        
        def calculate_message_priority(message, author):
            """Calculate message priority for processing order."""
            priority = 0
            
            # Longer messages get higher priority
            priority += min(len(message) / 10, 5)
            
            # Questions get higher priority
            if '?' in message:
                priority += 3
            
            # Direct mentions or commands get highest priority
            if message.lower().startswith(('@', '!', '/')):
                priority += 5
            
            # Moderator/VIP messages get higher priority (if available in author data)
            if 'mod' in author.lower() or 'vip' in author.lower():
                priority += 2
            
            return priority
        
        # Initialize optimized LiveChat controller
        try:
            livechat_controller = LiveChatController(livechat_config)
            logger.info("Optimized LiveChat controller initialized")
        except Exception as e:
            logger.error(f"Failed to initialize optimized LiveChat: {e}")
            # Fallback to standard implementation
            livechat_process_worker(
                live_chat_queue, terminate_event, immediate_livechat_fetch_event, shared_config
            )
            return
        
        # Optimized message processing
        fetch_interval = livechat_config.get("fetch_interval", 0.5)  # Faster fetching
        batch_size = livechat_config.get("batch_size", 10)
        
        def optimized_message_processor():
            """Optimized message processing with batching."""
            last_fetch_time = 0
            
            while not terminate_event.is_set():
                try:
                    current_time = time.time()
                    
                    if (immediate_livechat_fetch_event.is_set() or 
                        (current_time - last_fetch_time) >= fetch_interval):
                        
                        # Fetch messages
                        raw_messages = livechat_controller.get_new_messages(limit=batch_size)
                        
                        if raw_messages:
                            # Process batch with filtering
                            processed_messages = process_message_batch(raw_messages)
                            
                            # Send processed messages to queue
                            for msg_data in processed_messages:
                                if terminate_event.is_set():
                                    break
                                
                                try:
                                    live_chat_queue.put_nowait((msg_data["message"], msg_data["context"]))
                                    logger.debug(f"Queued priority message: {msg_data['message'][:30]}...")
                                except:
                                    # Handle full queue
                                    try:
                                        live_chat_queue.get_nowait()
                                        live_chat_queue.put_nowait((msg_data["message"], msg_data["context"]))
                                    except:
                                        logger.warning(f"Dropped message due to full queue: {msg_data['message'][:30]}...")
                        
                        last_fetch_time = current_time
                        immediate_livechat_fetch_event.clear()
                    
                    time.sleep(0.05)  # Faster polling
                
                except Exception as e:
                    logger.error(f"Error in optimized message processor: {e}")
                    time.sleep(1)
        
        # Start optimized processor
        processor_thread = threading.Thread(target=optimized_message_processor, daemon=True)
        processor_thread.start()
        
        logger.info("Optimized LiveChat worker ready")
        
        # Main thread
        while not terminate_event.is_set():
            time.sleep(0.1)
        
        processor_thread.join(timeout=2)
    
    except Exception as e:
        logger.error(f"Error in optimized LiveChat worker: {e}", exc_info=True)
        # Fallback to standard implementation
        livechat_process_worker(
            live_chat_queue, terminate_event, immediate_livechat_fetch_event, shared_config
        )
    finally:
        logger.info("Optimized LiveChat worker shutting down...")