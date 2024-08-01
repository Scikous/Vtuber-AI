import json
import numpy as np
import re

class LLMUtils:
    @staticmethod
    def get_rand_token_len(min_tokens=15, max_tokens=100, input_len=0):
        """
        Given an input (STT/Comment), the potential response length should have a higher chance of being longer.
        """
        # Adjust max tokens based on input length to avoid cutting off mid-thought
        adjusted_max_tokens = max(min_tokens, max_tokens - input_len)
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
    def load_model_exllamav2(model_dir="LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"):
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

        config = ExLlamaV2Config(model_dir)
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len = 65536, lazy = True)
        model.load_autosplit(cache, progress = True)
        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2DynamicGenerator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
        )
        #default text generation settings, can be overridden
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature = 0.9, 
            top_p = 0.8,
            token_repetition_penalty = 1.025
        )
        return generator, gen_settings, tokenizer
    
    #legacy model loading
    @staticmethod
    def load_model(base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                        device_map="auto",
                                                        trust_remote_code=False,
                                                        revision="main")

        if custom_model_name:
            print(custom_model_name)
            config = PeftConfig.from_pretrained(custom_model_name)
            model = PeftModel.from_pretrained(
                model, custom_model_name, offload_folder="LLM/offload")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        return model, tokenizer
    
    @staticmethod
    def load_character(character_info_json=""):
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