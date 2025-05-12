from model_utils import LLMUtils

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
    def load_model_exllamav2(cls, model_dir="LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ", character_name='assistant'):
        """
        Loads an ExLlamaV2 compatible model

        Returns:
        
            generator (ExLlamaV2DynamicGenerator): to be used for text generation

            gen_settings (ExLlamaV2Sampler.Settings): default text generation settings -- reasonably unique responses
            
            tokenizer (ExLlamaV2Tokenizer): the initialized tokenizer -- given to the "model" in models.py
        """
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
            temperature = 1.97, 
            top_p = 0.95,
            min_p=0.05,
            token_repetition_penalty = 1.035
        )
        return cls(generator, gen_settings, tokenizer, character_name)


    async def dialogue_generator(self, prompt, PromptTemplate, max_tokens=200,):
        """
        Generates character's response to a given input (Message)

        For text length variety's sake, randomly selects token length to appear more natural
        """
        # print(self.tokenizer.encode("instructions", encode_special_tokens = False, add_bos = False))
        # print("WHAHAHSSHSHS:\n\n", self.tokenizer.encode(prompt, encode_special_tokens = True, add_bos = False), self.tokenizer.bos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token, self.tokenizer.eos_token_id)

        prompt = PromptTemplate(user_str=prompt)
        # print("HAAAAA:\n", prompt, self.tokenizer.encode(prompt, encode_special_tokens = True, add_bos = False), self.tokenizer.bos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token, self.tokenizer.eos_token_id)

        max_tokens = LLMUtils.get_rand_token_len(max_tokens=max_tokens)
        #prompt = ["Five good reasons to adopt a cat:","Tell 5 simple jokes:", "how much is 8 + 19?"],
        # print(prompt)
        output = self.generator.generate(
            prompt = prompt,
            encode_special_tokens=True,
            decode_special_tokens=True,
            completion_only=True,
            max_new_tokens = max_tokens,
            stop_conditions = [self.tokenizer.eos_token_id],
            gen_settings = self.gen_settings,
            add_bos = False)
        # output = LLMUtils.character_reply_cleaner(output, self.character_name)
        return output

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

    async def dialogue_generator(self, prompt, PromptTemplate):
        """
        Generates character's response to a given input (Message)
        """
        def is_incomplete_sentence(text):
            return text.strip()[-1] not in {'.', '!', '?'}

        max_attempts = 7
        prompt = PromptTemplate(user_str=prompt)
        comment_tokenized = self.tokenizer(prompt, return_tensors="pt")
        print("HAAAAA:\n",comment_tokenized, self.tokenizer.bos_token, self.tokenizer.bos_token_id, self.tokenizer.eos_token, self.tokenizer.eos_token_id)
        inputs = self.tokenizer(prompt, return_tensors="pt")

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