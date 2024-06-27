# import json
# import numpy as np
# import re
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments, pipeline
# # import datasets
# # from difflib import SequenceMatcher
# # responses are shorter or longer to create a more natural way of responding
# def get_rand_token_len(min_tokens=15, max_tokens=100, input_len=0):
#     """
#     WIP
#     given an input (STT/Comment), the potential response length should have a higher chance of being longer.  
#     """
#     # Adjust max tokens based on input length to avoid cutting off mid-thought
#     adjusted_max_tokens = max(min_tokens, max_tokens - input_len)
#     print(adjusted_max_tokens)
#     tokens = np.arange(min_tokens, adjusted_max_tokens)
#     token_weights = np.linspace(
#         start=1.0, stop=0.05, num=adjusted_max_tokens-min_tokens)
#     token_weights /= np.sum(token_weights)
#     token_len = np.random.choice(tokens, p=token_weights)
#     # print("TOKEENSS", token_len, tokens)
#     return token_len

# def model_loader(base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
#     model = AutoModelForCausalLM.from_pretrained(base_model_name,
#                                                 device_map="auto",
#                                                 trust_remote_code=False,
#                                                 # offload="LLM/offload",
#                                                 revision="main")

#     if custom_model_name:
#         print(custom_model_name)
#         config = PeftConfig.from_pretrained(custom_model_name)
#         model = PeftModel.from_pretrained(
#             model, custom_model_name, offload_folder="LLM/offload")
#         # config.init_lora_weights = False
#         # model.add_adapter(peft_config=config)
#         # model.enable_adapters()

#     model.eval()
#     # load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
#     return model, tokenizer


# def character_loader(character_info_json=""):
#     if character_info_json:
#         with open(character_info_json, 'r') as character:
#             character_info = json.load(character)
#             instructions, user_name, character_name = character_info[
#                 "instructions"], character_info["user_name"], character_info["character_name"]
#     else:
#         instructions, user_name, character_name = "", "user", "assistant"
#     return instructions, user_name, character_name

# #remove words after last sentence stopper (., ?, !)
# def sentence_reducer(output_clean):
#     # Find the last occurrence of a sentence stopper
#     match = re.search(r'[.!?](?!.*[.!?])', output_clean)
#     if match:
#         # Position of the last sentence stopper
#         pos = match.end()
#         #print("I AM EHEREREE"*10, pos)
#         # Truncate the text at the position after the last stopper
#         output_clean = output_clean[:pos].strip()
#     return output_clean

# class VtuberLLM():
#     def __init__(self, model, tokenizer, character_name):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.character_name = character_name

#     # generates character's response to a given input (TTS/Comment)
#     def dialogue_generator(self, comment, PromptTemplate):
#         # generated text may end prematurely, this should help avoid that
#         def is_incomplete_sentence(text):
#             # reply will end on not ., !, ? clean it beforehand then check
#             return text.strip()[-1] not in {'.', '!', '?'}

#         max_attempts = 7  # Set a limit to avoid infinite loops
#         # continuation_prompt = "Continue from where the text within context left off. It is IMPERATIVE that you do NOT repeat anything already mentioned in the context. YOU simply just continue the previous without copying it."

#         prompt = PromptTemplate(user_str=comment)
#         comment_tokenized = self.tokenizer(comment, return_tensors="pt")
#         inputs = self.tokenizer(prompt, return_tensors="pt")

#         generated_text = ""
#         for attempt in range(max_attempts):
#             max_new_tokens = get_rand_token_len(
#                 input_len=len(comment_tokenized["input_ids"][0]))
#             results = self.model.generate(input_ids=inputs["input_ids"].to(
#                 "cuda"), max_new_tokens=max_new_tokens, top_p=0.8, top_k=50, temperature=1.1, repetition_penalty=1.2, do_sample=True)
#             output = self.tokenizer.batch_decode(results, skip_special_tokens=True)[0]
#             # print(output)
#             # only add new unique responses to final output
#             print(f"{'#'*30}\n{output}\n{'#'*30}")
#             output_clean = self.character_reply_cleaner(output).lower()
#             print(f"{'#'*30}\n{output_clean}\n{'#'*30}")
#             generated_text = output_clean
#             # if attempt == 0:
#             #     generated_text = output_clean
#             #     gen_texts.append(generated_text)
#             # else:
#             #     similarity = SequenceMatcher(None, gen_texts[-1], output_clean).ratio()
#             #     if similarity >= 0.4:
#             #         new_text_incomplete = is_incomplete_sentence(output_clean)
#             #         if not new_text_incomplete:
#             #             gen_texts[-1] = output_clean
#             #             generated_text = " ".join(gen_texts)
#             #             print("*"*30,"\n\n",generated_text, "\n\n", "*"*30)
#             #     else:
#             #         generated_text = generated_text + ' ' + output_clean #add space after each generated response #character_reply_cleaner(copy.copy(output)).lower() #gen-text only needs actual response(s)
#             #         gen_texts.append(output_clean)
#             # print("WHEHEEE"*15, generated_text)
#             if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
#                 break  # Break if the sentence is complete or max attempts reached

#             # # Prepare for continuation
#             # prompt = PromptTemplate(user_str=continuation_prompt, context_str=generated_text)
#             # #print(prompt)
#             # inputs = tokenizer(prompt, return_tensors="pt")
#         # print(f"\n\n\nDKSDOSOJ{generated_text}\n\n\n")

#         print("Text generation finished")
#         return generated_text


#     def character_reply_cleaner(self, reply):
#         character_name = self.character_name+'\n'#"john\n"
#         character_index = reply.find(character_name)

#         if character_index != -1:
#             reply = reply[character_index + len(character_name):]
#         else:
#             print("Womp womp", reply)
#         # print("w"*30, '\n\n', reply, '\n\n',"w"*30, '\n\n')

#         #########################################################
#         # try:
#         #     pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?=\<\/|\<|im_end\||$)"

#         #     match = re.search(pattern, reply, re.DOTALL)
#         #     clean_reply = match.group(0).strip() if match else "Womp Womp"
#         #     if clean_reply == "Womp Womp":
#         #         raise ValueError(" Womp Womp")
#         #     print("hmmm")
#         # except ValueError:  # not sure if getting triggered ever.
#         #     pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?:\n|$)"
#         #     match = re.search(pattern, reply, re.DOTALL)
#         #     clean_reply = match.group(1).strip() if match else "Womp Womp sequel"
#         #     print('ffff', clean_reply, 'ffffff')
#         ##########################################################
#         reply = sentence_reducer(reply)
#         return reply




import json
import numpy as np
import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler

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
    def load_model_exllama(base_model_name="LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
        model_dir = "LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"
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
    @staticmethod
    def load_model(base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
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
    

    #####################
    #deprecated?
    # @staticmethod
    # def convert_model_to_Onnx(model_name):
    #     base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
    #     model = AutoModelForCausalLM.from_pretrained(base_model_name,
    #                                                  device_map="cuda",
    #                                                  trust_remote_code=False,
    #                                                  revision="main"
    #                                                  #attn_implementation="eager"
    #                                                  )
    #     config = PeftConfig.from_pretrained(model_name)
    #     model = PeftModel.from_pretrained(
    #     model, model_name , offload_folder="LLM/offload")

    #     # Load your model and tokenizer
    #     #model = LLMUtils.load_model(custom_model_name=model_name)
    #     #model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
    #     #model = AutoModelForCausalLM.from_pretrained(model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_name)


    #     # Dummy input for the model (required for export)
    #     dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")
    #     dummy_input = dummy_input.to("cuda")
    #     # ort_model = ORTModelForSequenceClassification.from_pretrained(model, export=True)
    #     # #ort_model = ort_model.to("cuda")
    #     # ort_model.save_pretrained("LLM/onnx")
    #     # dummy_input = torch.tensor([tokenizer.encode("Hello, how are you?")])
    #     # dummy_input = dummy_input.to("cuda")
    #     # attention_mask = torch.ones((1, len(tokenizer.encode("Hello, how are you?"))))
    #     # attention_mask = attention_mask.to("cuda")
    #     # Trace the model with the attention mask
    #     # traced_model = torch.jit.trace(model, (dummy_input, attention_mask))
    #     # Export the model to ONNX
    #     attention_mask = dummy_input["attention_mask"]
    #     attention_mask = attention_mask.to("cuda")

    #     # torch.onnx.export(model, (dummy_input["input_ids"], attention_mask), "LLM/model.onnx",
    #     #           input_names=["input_ids", "attention_mask"], output_names=["output"], opset_version=18)
    #     # torch.onnx.dynamo_export(model, dummy_input, "tmodel.onnx")
    #     # try:
    #     #     with torch.no_grad():
    #     #     print("Model successfully exported to ONNX.")
    #     # except torch.onnx.OnnxExporterError as e:
    #     #     print(f"Failed to export the model to ONNX. Error: {e}")
    #     #     with open("report_dynamo_export.sarif", "r") as file:
    #     #         sarif_report = file.read()
    #     #     print(sarif_report)

#current, lowest latency
class VtuberExllamav2:
    def __init__(self, generator, gen_settings, tokenizer, character_name):
        self.generator = generator
        self.gen_settings = gen_settings
        self.tokenizer = tokenizer
        self.character_name = character_name

    def dialogue_generator(self, prompt, PromptTemplate, max_tokens=200):
        prompt = PromptTemplate(user_str=prompt)
        #prompt = ["Five good reasons to adopt a cat:","Tell 5 simple jokes:", "how much is 8 + 19?"],
        output = self.generator.generate(
            prompt = prompt,
            encode_special_tokens=True,
            max_new_tokens = max_tokens,
            stop_conditions = [self.tokenizer.eos_token_id],
            gen_settings = self.gen_settings,
            add_bos = True)
        output = self.character_reply_cleaner(output)
        return output
    def character_reply_cleaner(self, reply):
        character_name = self.character_name + '\n'
        character_index = reply.find(character_name)

        if character_index != -1:
            reply = reply[character_index + len(character_name):]
        else:
            print("Womp womp", reply)
            
        reply = LLMUtils.sentence_reducer(reply)
        return reply

#legacy model, high latency
class VtuberLLM:
    def __init__(self, model, tokenizer, character_name):
        self.model = model
        self.tokenizer = tokenizer
        self.character_name = character_name

    def dialogue_generator(self, comment, PromptTemplate):
        """
        Generates character's response to a given input (TTS/Comment)
        """
        def is_incomplete_sentence(text):
            return text.strip()[-1] not in {'.', '!', '?'}

        max_attempts = 7
        prompt = PromptTemplate(user_str=comment)
        comment_tokenized = self.tokenizer(comment, return_tensors="pt")
        inputs = self.tokenizer(prompt, return_tensors="pt")

        generated_text = ""
        for attempt in range(max_attempts):
            max_new_tokens = LLMUtils.get_rand_token_len(input_len=len(comment_tokenized["input_ids"][0]))
            results = self.model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                          max_new_tokens=max_new_tokens,
                                          top_p=0.8, top_k=50, temperature=1.1,
                                          repetition_penalty=1.2, do_sample=True)
            output = self.tokenizer.batch_decode(results, skip_special_tokens=True)[0]
            print(f"{'#' * 30}\n{output}\n{'#' * 30}")
            output_clean = self.character_reply_cleaner(output).lower()
            print(f"{'#' * 30}\n{output_clean}\n{'#' * 30}")
            generated_text = output_clean
            if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
                break

        print("Text generation finished")
        return generated_text

    def character_reply_cleaner(self, reply):
        character_name = self.character_name + '\n'
        character_index = reply.find(character_name)

        if character_index != -1:
            reply = reply[character_index + len(character_name):]
        else:
            print("Womp womp", reply)
            
        reply = LLMUtils.sentence_reducer(reply)
        return reply
