from model_utils import LLMUtils
#current, lowest latency
class VtuberExllamav2:
    def __init__(self, generator, gen_settings, tokenizer, character_name):
        self.generator = generator
        self.gen_settings = gen_settings
        self.tokenizer = tokenizer
        self.character_name = character_name

    async def dialogue_generator(self, prompt, PromptTemplate, max_tokens=200):
        # prompt = PromptTemplate(user_str=prompt)
        max_tokens = LLMUtils.get_rand_token_len(max_tokens=max_tokens)
        #prompt = ["Five good reasons to adopt a cat:","Tell 5 simple jokes:", "how much is 8 + 19?"],
        output = self.generator.generate(
            prompt = prompt,
            encode_special_tokens=True,
            decode_special_tokens=True,
            completion_only=True,
            max_new_tokens = max_tokens,
            stop_conditions = [self.tokenizer.eos_token_id],
            completion_only=True,
            gen_settings = self.gen_settings,
            add_bos = True)
        # output = LLMUtils.character_reply_cleaner(output, self.character_name)
        return output

#legacy model, high latency
class VtuberLLM:
    def __init__(self, model, tokenizer, character_name):
        self.model = model
        self.tokenizer = tokenizer
        self.character_name = character_name

    def dialogue_generator(self, prompt, PromptTemplate):
        """
        Generates character's response to a given input (TTS/Comment)
        """
        def is_incomplete_sentence(text):
            return text.strip()[-1] not in {'.', '!', '?'}

        max_attempts = 7
        prompt = PromptTemplate(user_str=prompt)
        comment_tokenized = self.tokenizer(prompt, return_tensors="pt")
        inputs = self.tokenizer(prompt, return_tensors="pt")

        generated_text = ""
        for attempt in range(max_attempts):
            max_new_tokens = LLMUtils.get_rand_token_len(input_len=len(comment_tokenized["input_ids"][0]))
            results = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                          max_new_tokens=max_new_tokens,
                                          top_p=0.8, top_k=50, temperature=1.1,
                                          repetition_penalty=1.2, do_sample=True)
            output = self.tokenizer.batch_decode(results, skip_special_tokens=True)[0]
            # print(f"{'#' * 30}\n{output}\n{'#' * 30}")
            output_clean = LLMUtils.character_reply_cleaner(output, self.character_name).lower()
            # print(f"{'#' * 30}\n{output_clean}\n{'#' * 30}")
            generated_text = output_clean
            if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
                break

        # print("Text generation finished")
        return generated_text