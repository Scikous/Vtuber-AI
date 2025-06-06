import json
import numpy as np
import re

def apply_chat_template(instructions, prompt, tokenizer, conversation_history=None):
    """
    Applies a chat template to the prompt with optional conversation history.
    
    Args:
        instructions: System instructions string
        prompt: Current user message string
        tokenizer: Tokenizer to use
        previous_conversation: List of previous messages in order [user_msg1, assistant_msg1, user_msg2, assistant_msg2, ...]
                            First message should be user, then alternating user/assistant
    """
    
    messages = [{"role": "system", "content": instructions}]
    
    # Add previous conversation history if provided
    if conversation_history:
        for i, msg in enumerate(conversation_history):
            # Alternate between user and assistant roles
            # Even indices (0, 2, 4...) are user messages
            # Odd indices (1, 3, 5...) are assistant messages
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg})
    
    # Add the current prompt as the latest user message
    messages.append({"role": "user", "content": prompt})
    print("FINAL MESSAGES", messages)
    tokenized_chat = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    return tokenized_chat

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

def get_rand_token_len(min_tokens=15, max_tokens=100, input_len=0):
    """
    Given an input (Message), the potential response length should have a higher chance of being longer.
    """
    # Adjust max tokens based on input length to avoid cutting off mid-thought
    adjusted_max_tokens = max(min_tokens, max_tokens - input_len)+1
    # print(adjusted_max_tokens)
    tokens = np.arange(min_tokens, adjusted_max_tokens)
    token_weights = np.linspace(
        start=1.0, stop=0.05, num=adjusted_max_tokens - min_tokens)
    token_weights /= np.sum(token_weights)
    token_len = np.random.choice(tokens, p=token_weights)
    return token_len

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


##deprecated -- legacy purposes only
def sentence_reducer(output_clean):
    """
    Remove words after the last sentence stopper (., ?, !)
    """
    match = re.search(r'[.!?](?!.*[.!?])', output_clean)
    if match:
        pos = match.end()
        output_clean = output_clean[:pos].strip()
    return output_clean

def character_reply_cleaner(reply, character_name):
    """
    Clean the character's reply by removing the character's name and truncating after the last sentence stopper.
    """
    character_name = character_name + '\n'
    character_index = reply.find(character_name)

    if character_index != -1:
        reply = reply[character_index + len(character_name):]
    else:
        print("Womp womp", reply)
        
    reply = sentence_reducer(reply)
    return reply

# def extract_name_message(input_string):
#     """
#     Extracts the name and message from a formatted string.

#     Args:
#         input_string (str): The formatted string containing a name and a message.
#                             E.g., "RestreamBot:[YouTube: Michael] hello"

#     Returns:
#         str: The extracted name and message in the format "Michael: hello"
#     """
#     # Use regular expression to match the pattern
#     match = re.search(r'\[\w+:\s?(.+?)\]\s?(.+)', input_string)

#     # match = re.search(r'\[(\w+:\s?\w+)\]\s?(.+)', input_string)
#     if match:

#         name, message = match.groups()
#         return f"{name.split(':', 1)[1].strip()}: {message.strip()}"

#     else:
#         return input_string  # Return the original string if no match is found
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
