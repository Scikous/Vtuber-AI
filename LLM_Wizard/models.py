from model_utils import LLMUtils
import torch
import gc
import asyncio
#current, lowest latency
class VtuberExllamav2:
    """
    Holds and handles all of the ExllamaV2 based tools
    """

    def __init__(self, generator, gen_settings, tokenizer, character_name):
        self.generator = generator
        self.gen_settings = gen_settings
        self.tokenizer = tokenizer
        self.character_name = character_name

    @classmethod
    def load_model_exllamav2(cls, model_dir="./LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ", character_name='assistant'):
        """
        Loads an ExLlamaV2 compatible model

        Returns:
        
            generator (ExLlamaV2DynamicGenerator): to be used for text generation

            gen_settings (ExLlamaV2Sampler.Settings): default text generation settings -- reasonably unique responses
            
            tokenizer (ExLlamaV2Tokenizer): the initialized tokenizer -- given to the "model" in models.py
        """
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicGeneratorAsync, ExLlamaV2Sampler
        from transformers import AutoTokenizer
        
        #transformers tokenizer, not exllamav2's tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir) # for applying chat template
        
        config = ExLlamaV2Config(model_dir)
        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, max_seq_len = 65536, lazy = True)
        model.load_autosplit(cache, progress = True)

        generator_async = ExLlamaV2DynamicGeneratorAsync(
            model = model,
            cache = cache,
            tokenizer = ExLlamaV2Tokenizer(config),
        )
        #default text generation settings, can be overridden
        gen_settings = ExLlamaV2Sampler.Settings(
            temperature = 1.97, 
            top_p = 0.95,
            min_p=0.05,
            token_repetition_penalty = 1.035
        )

        return cls(generator_async, gen_settings, tokenizer, character_name)

    def __del__(self):
        # Manual cleanup (if necessary)
        del self.generator
        del self.gen_settings
        del self.tokenizer
        torch.cuda.empty_cache()  # If using CUDA
        gc.collect()
        print("Deleted and garbage collected ExllamaV2 Model!")

    async def dialogue_generator(self, prompt, max_tokens=200):
        """
        Generates character's response to a given input (Message)

        For text length variety's sake, randomly selects token length to appear more natural
        """
        from exllamav2.generator import ExLlamaV2DynamicJobAsync
        prompt = LLMUtils.apply_chat_template(instructions="", prompt=prompt,tokenizer=self.tokenizer)

        max_tokens = LLMUtils.get_rand_token_len(max_tokens=max_tokens)
        async_job = ExLlamaV2DynamicJobAsync(
                        generator=self.generator,
                        encode_special_tokens=False,
                        decode_special_tokens=False,
                        completion_only=True,
                        input_ids=prompt,
                        # input_ids=self.tokenizer.encode("foools"),
                        max_new_tokens=max_tokens,
                        #     stop_conditions = [self.tokenizer.eos_token_id],
                        gen_settings=self.gen_settings,
                        add_bos = False #if using apply_chat_template set to false -- only plain string should have True)
                        #token_healing = False #True if output is weird, False if output is un-weird
                        #return_logits = False #for analyzing model's probability distribution before sapling -- generally don't touch
                        #return_probs = False #for understanding the model's confidence in its choices -- generally don't touch
                        #filters = None #list[list[ExLlamaV2Filter]] | list[ExLlamaV2Filter] forcing/guiding text generation
                        #identifier = None #object for tracking/associating metadata
                        #banned_strings = None #list[str] for banning specific words/phrases
                        #embeddings = None #list[ExLlamaV2MMEmbedding] can input images thathave been embedded into vectors

                    )
        return async_job    

#legacy model, high latency
class VtuberLLM:
    def __init__(self, model, tokenizer, character_name):
        self.model = model
        self.tokenizer = tokenizer
        self.character_name = character_name

    @classmethod
    def load_model(cls, base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name="", character_name='assistant'):
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
        return cls(model, tokenizer, character_name)

    async def dialogue_generator(self, prompt):
        """
        Generates character's response to a given input (Message)
        """
        def is_incomplete_sentence(text):
            return text.strip()[-1] not in {'.', '!', '?'}

        max_attempts = 7
        comment_tokenized = LLMUtils.apply_chat_template(instructions="", prompt=prompt,tokenizer=self.tokenizer)
        print("HAAAAA:\n",comment_tokenized, self.tokenizer.bos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        inputs = LLMUtils.apply_chat_template(instructions="", prompt=prompt,tokenizer=self.tokenizer)

        generated_text = ""
        print(len(comment_tokenized["input_ids"][0]))
        for attempt in range(max_attempts):
            max_new_tokens = LLMUtils.get_rand_token_len(input_len=len(comment_tokenized["input_ids"][0]))
            results = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                          max_new_tokens=max_new_tokens,
                                          top_p=0.8, top_k=50, temperature=1.1,
                                          repetition_penalty=1.2, do_sample=True, num_return_sequences=10)
            output = self.tokenizer.batch_decode(results, skip_special_tokens=True)[0]
            # print(f"{'#' * 30}\n{output}\n{'#' * 30}")
            output_clean = LLMUtils.character_reply_cleaner(output, self.character_name).lower()
            # print(f"{'#' * 30}\n{output_clean}\n{'#' * 30}")
            generated_text = output_clean
            if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
                break

        # print("Text generation finished")
        return generated_text