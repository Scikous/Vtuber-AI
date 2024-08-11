# # from model_utils import LLMUtils
# custom_model = "LLM/unnamedSICUACCT"

# # from exllamav2 import ExLlamaV2
# # from transformers import AutoTokenizer

# # from exllamav2.generator import (
# #     ExLlamaV2Sampler,
# # )


# # model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
# # #model, tokenizer = load_exllamav2_model(model_name)


# # settings = ExLlamaV2Sampler.Settings()
# # settings.temperature = 0.85
# # settings.top_k = 50
# # settings.top_p = 0.8
# # settings.token_repetition_penalty = 1.05


# # model = ExLlamaV2(model_path=model_name, settings=settings, streaming=True, max_new_tokens=150, verbose=True)
# # tokenizer = AutoTokenizer.from_pretrained(model_name)

# # # LLMUtils.convert_model_to_Onnx(custom_model)



# from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
# import time
# model_dir = "LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"
# config = ExLlamaV2Config(model_dir)
# model = ExLlamaV2(config)
# cache = ExLlamaV2Cache(model, max_seq_len = 65536, lazy = True)
# model.load_autosplit(cache, progress = True)
# tokenizer = ExLlamaV2Tokenizer(config)

# from exllamav2.generator import ExLlamaV2DynamicGenerator
# from llm_templates import PromptTemplate

# generator = ExLlamaV2DynamicGenerator(
#     model = model,
#     cache = cache,
#     tokenizer = tokenizer,
# )

# pt = PromptTemplate(instructions_str="John Smith also known as john takes everything to an extreme. He dresses in outlandish and flamboyant clothing, often with theatricality. His speech is full of bombastic, grand and dramatic pronouncements. You must read the context first carefully before responding. You must name the drink you've drunk if asked.", user_name="user", character_name="john")
# prompt = pt.capybaraChatML(user_str="What did you drink today?",context_str="context: john: mymymy it's already nighttime\nuser:so it seems.\njohn:The coca cola was great though!")
# # prompt = pt.capybaraChatML(user_str="context: john: mymymy it's already nighttime\nuser:so it seems.\njohn:The coca cola was great though!\n\nWhat did you drink today?",context_str="")
# # prompt = f"""John Smith also known as john takes everything to an extreme. He dresses in outlandish and flamboyant clothing, often with theatricality. His speech is full of bombastic, grand and dramatic pronouncements. You must read the context first carefully before responding. You must name the drink you've drunk if asked.\n\ncontext: john: mymymy it's already nighttime\nuser:so it seems.\njohn:The coca cola was great though!\n\nWhat did you drink today?"""
# s = time.time()
# satr = time.perf_counter()
# from exllamav2.generator import ExLlamaV2Sampler
# gen_settings = ExLlamaV2Sampler.Settings(
#     temperature = 0.9, 
#     top_p = 0.8,
#     token_repetition_penalty = 1.025
# )

# for _ in range(5):
#     output = generator.generate(
#         prompt = prompt,
#         encode_special_tokens=True,
#         max_new_tokens = 400,
#         stop_conditions = [tokenizer.eos_token_id],
#         gen_settings = gen_settings,
#         add_bos = True)
#     print(output+'\n\n')
# end = time.perf_counter()
# e = time.time()
# # print(output,satr,end, end-satr, e-s)

# # def vllm_test():
# #     from vllm import LLM, SamplingParams
# #     prompts = [
# #     "Hello, my name is",
# #     "The president of the United States is",
# #     "The capital of France is",
# #     "The future of AI is",
# #     ]
# #     sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# #     llm = LLM(model=model_dir)
# #     outputs = llm.generate(prompts, sampling_params)

# # # Print the outputs.
# #     for output in outputs:
# #         prompt = output.prompt
# #         generated_text = output.outputs[0].text
# #         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

from models import VtuberExllamav2

from model_utils import LLMUtils
from llm_templates import PromptTemplate as pt
from time import perf_counter

import asyncio

character_info_json = "LLM/characters/character.json"
instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

instructions_string = f"""{instructions}"""
PromptTemplate = pt(instructions_string, user_name, character_name)
# custom_model = "LLM/unnamedSICUACCT"
# Character = VtuberLLM.load_model(custom_model=custom_model, character_name=character_name)

#exllamav2 model
# generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2() #deprecated
Character = VtuberExllamav2.load_model_exllamav2(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)


dummy_data = ["Good day, state your name.", "What is your favorite drink?", "Do you edge?"]

# custom_model = "LLM/unnamedSICUACCT"
# model, tokenizer = LLMUtils.load_model(custom_model_name=custom_model)
# Character = VtuberLLM(model, tokenizer, character_name)  

# start = perf_counter()
# response = Character.dialogue_generator(prompt=dummy_data[0], PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=400)
# response1 = Character.dialogue_generator(prompt=dummy_data[1], PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=400)
# response2 = Character.dialogue_generator(prompt=dummy_data[2], PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=400)
# end = perf_counter()

# print(f"Prompts: {dummy_data}\nResponses:\n{response}\n\n{response1}\n\n{response2}\n\nTime Taken (Seconds): {end-start}")

msg = """You MUST have the following in your output EXACTLY as written: "hello", 'wow't'"""
start = perf_counter()
response = asyncio.run(Character.dialogue_generator(prompt=msg, PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=400))
end = perf_counter()

print(f"Prompts: {dummy_data}\nResponses:\n{response}\n\nTime Taken (Seconds): {end-start}")