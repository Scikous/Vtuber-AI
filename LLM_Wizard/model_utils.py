import json
import numpy as np
import re

import asyncio
import io
import logging
import os
from typing import List, Optional

import aiohttp
from PIL import Image
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)

def apply_chat_template(
    instructions: str,
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    conversation_history: Optional[List[str]] = None,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False
):
    """
    Applies a chat template to the prompt with optional conversation history.
    
    Args:
        instructions: System instructions string
        prompt: Current user message string
        tokenizer: Tokenizer to use
        conversation_history: List of previous messages in order [user_msg1, assistant_msg1, user_msg2, assistant_msg2, ...]
                            First message should be user, then alternating user/assistant
        tokenize: Whether to tokenize the output
        add_generation_prompt: Whether to add a generation prompt
        continue_final_message: Whether to treat the prompt as a continuation of the assistant's message
    """
    messages = [{"role": "system", "content": instructions}]
    
    if conversation_history:
        for i, msg in enumerate(conversation_history):
            # even = user, odd = assistant
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg})
    
    # continue from last LLM generation or create brand new generation
    if continue_final_message:
        messages.append({"role": "assistant", "content": prompt})
    else:
        messages.append({"role": "user", "content": prompt})
    
    # apply_chat_template handles BOS token logic internally when tokenize=True
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=tokenize, 
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        return_tensors="pt"
    )



def prompt_wrapper(message, context=""):
    from textwrap import dedent

    """
    Wraps the user message and context within the correct styled prompt.

    IS NOT APPLYING CHAT TEMPLATE -- SEE apply_chat_template() for that
    """
    prompt = dedent(
        f"""
    {message}

    Information:

    ```
    {context}
    ```
    """
    )
    return prompt

def load_character(character_info_json=""):
    """
    Returns:
        instructions (str): tells how the LLM (character) should behave
        user_name (str): the name of the user (probably deleted in the future)
        character_name (str): the name of the character
    """
    if character_info_json:
        with open(character_info_json, 'r') as character:
            character_info = json.load(character)
            instructions = character_info["instructions"]
            user_name = character_info["user_name"]
            character_name = character_info["character_name"]
    else:
        instructions, user_name, character_name = "", "user", "assistant"
    return instructions, user_name, character_name

def contains_sentence_terminator(chunk_text, sentence_terminators=['.', '!', '?', ',']):
    """
    Checks if the given text chunk contains any of the specified sentence terminators.

    Args:
        chunk_text (str): The text chunk to check.
        sentence_terminators (list): A list of characters that denote the end of a sentence.

    Returns:
        bool: True if a sentence terminator is found in the chunk_text, False otherwise.
    """
    for terminator in sentence_terminators:
        if terminator in chunk_text:
            return True
    return False


def extract_name_message(input_string):
    """
    Extracts the name and message from a formatted string.

    Args:
        input_string (str): The formatted string containing a name and a message.
                            E.g., "RestreamBot:[YouTube: Michael] hello"

    Returns:
        str: The extracted name and message in the format "Michael: hello"
    """
    # New regular expression:
    # r'\[(\w+):\s*(.+?)\]\s*(.+)'
    # Breakdown:
    # \[              - Matches the literal opening square bracket
    # (\w+)           - Group 1: Captures the platform/channel type (e.g., "YouTube")
    # :               - Matches the literal colon
    # \s* - Matches zero or more whitespace characters
    # (.+?)           - Group 2 (non-greedy): Captures the name (allowing spaces)
    # \]              - Matches the literal closing square bracket
    # \s* - Matches zero or more whitespace characters before the message
    # (.+)            - Group 3: Captures the message
    match = re.search(r'\[(\w+):\s*(.+?)\]\s*(.+)', input_string)
    if match:
        # channel_type is match.group(1) (e.g., "YouTube")
        extracted_name = match.group(2).strip() # This is the name we want (e.g., "zaza dznelashvili")
        message = match.group(3).strip()       # This is the message

        return f"{extracted_name}: {message}"
    else:
        # If no match, return the original string or handle as appropriate
        return input_string


#synchronous get_image
# def get_image(file = None, url = None):
#     assert (file or url) and not (file and url)
#     if file:
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         file_path = os.path.join(script_dir, file)
#         return Image.open(file_path)
#     elif url:
#         return Image.open(requests.get(url, stream = True).raw)


async def get_image(file: Optional[str] = None, url: Optional[str] = None) -> Image.Image:
    """
    Asynchronously loads an image from a local file path or a URL.

    Args:
        file: The local file name (relative to this script's directory).
        url: The URL of the image to download.

    Returns:
        A PIL Image object.

    Raises:
        ValueError: If both 'file' and 'url' are provided, or if neither is.
        FileNotFoundError: If the local file does not exist.
        Exception: For network-related errors during download.
    """
    if not (file or url) or (file and url):
        raise ValueError("Provide either a 'file' or a 'url', but not both.")

    if file:
        def _load_from_disk():
            # This blocking I/O function will be run in a separate thread
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found at: {file_path}")
            return Image.open(file_path)
        
        try:
            # Offload the blocking file I/O to a thread pool
            return await asyncio.to_thread(_load_from_disk)
        except Exception as e:
            log.error(f"Failed to load image from file '{file}': {e}")
            raise

    elif url:
        try:
            # Use an async HTTP client to avoid blocking the event loop
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data))
        except aiohttp.ClientError as e:
            log.error(f"Failed to download image from URL '{url}': {e}")
            raise