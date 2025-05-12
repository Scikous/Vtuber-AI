import json
import numpy as np
import re

class LLMUtils:
    @staticmethod
    def get_rand_token_len(min_tokens=15, max_tokens=100, input_len=0):
        """
        Given an input (Message), the potential response length should have a higher chance of being longer.
        """
        # Adjust max tokens based on input length to avoid cutting off mid-thought
        adjusted_max_tokens = max(min_tokens, max_tokens - input_len)+1
        print(adjusted_max_tokens)
        tokens = np.arange(min_tokens, adjusted_max_tokens)
        token_weights = np.linspace(
            start=1.0, stop=0.05, num=adjusted_max_tokens - min_tokens)
        token_weights /= np.sum(token_weights)
        token_len = np.random.choice(tokens, p=token_weights)
        return token_len

    @staticmethod
    def sentence_reducer(output_clean):
        """
        Remove words after the last sentence stopper (., ?, !)
        """
        match = re.search(r'[.!?](?!.*[.!?])', output_clean)
        if match:
            pos = match.end()
            output_clean = output_clean[:pos].strip()
        return output_clean
    
    @staticmethod
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
    
    @staticmethod
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
            
        reply = LLMUtils.sentence_reducer(reply)
        return reply
    