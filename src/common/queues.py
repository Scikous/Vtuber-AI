"""
Shared Queues for Vtuber-AI
Defines asyncio and multiprocessing queues used for communication between different components and services.
"""
import asyncio
from multiprocessing import Queue as MPQueue

# Asynchronous queues for intra-process communication (e.g., between async workers)
speech_queue = asyncio.Queue(maxsize=1)
live_chat_queue = asyncio.Queue(maxsize=1)
llm_output_queue = asyncio.Queue(maxsize=1)

# Multiprocessing queue for inter-process communication (e.g., for live chat process)
# This specific queue is intended for the live_chat_process to send messages to the main process.
mp_live_chat_message_queue = MPQueue()

# Potentially other queues can be added here as needed, for example:
# tts_input_queue = asyncio.Queue(maxsize=1) # If TTS worker takes text input via a dedicated queue
# audio_output_queue = asyncio.Queue() # For processed audio data ready for playback

# It's good practice to provide functions to get queue instances if more complex initialization is needed,
# but for simple global queues, direct definition is often sufficient for smaller applications.

def get_speech_queue():
    return speech_queue

def get_live_chat_queue():
    return live_chat_queue

def get_llm_output_queue():
    return llm_output_queue

def get_mp_live_chat_message_queue():
    return mp_live_chat_message_queue

# Example of how you might want to group them if they were part of a class
# class AppQueues:
#     def __init__(self):
#         self.speech_queue = asyncio.Queue(maxsize=1)
#         self.live_chat_queue = asyncio.Queue(maxsize=1)
#         self.llm_output_queue = asyncio.Queue(maxsize=1)
#         self.mp_live_chat_message_queue = MPQueue()

# app_queues = AppQueues() # Then use app_queues.speech_queue etc.