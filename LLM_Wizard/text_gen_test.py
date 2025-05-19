from models import VtuberExllamav2, VtuberLLM

from model_utils import LLMUtils
from llm_templates import PromptTemplate as pt
from time import perf_counter
import asyncio

# from general_utils import read_messages_csv

#get current character's information to use
character_info_json = "LLM_Wizard/characters/character.json"
instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

#set prompt template to follow current character's instruction set and name
instructions_string = f"""{instructions}"""
PromptTemplate = pt(instructions_string, user_name, character_name)
dummy_data = ["Good day, state your name.", "What is your favorite drink?", "Do you edge?"]

#LLM model to use
model = "LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model) # Example

# For tokenized output ready for model.generate()

# For string output (e.g., for debugging)
# formatted_chat_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(formatted_chat_string)

#test using the exllamav2
async def exllamav2_test():
    Character = VtuberExllamav2.load_model_exllamav2(model_dir=model,character_name=character_name)#(generator, gen_settings, tokenizer, character_name)

    start = perf_counter()
    # messages = [
    #     {"role": "system", "content": "You are a helpful AI assistant."},
    #     {"role": "user", "content": "Hello, what's the weather like today?"},
    #     # Add previous assistant messages if it's an ongoing conversation
    # ]
    # tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    #msg = "<|begin_of_text|>instructions: John Smith takes everything to an extreme. He dresses in outlandish and flamboyant clothing, often with theatricality. His speech is full of bombastic, grand and dramatic pronouncements. Everytime he speaks he says and responds ALWAYS with something NEW and UNIQUE EVERY SINGLE TIME. You will be penalized for saying anything present in this paragraph, and for not saying random unique things. He speaks of many fascinating topics.\n\n user message: Kid Flasher: Elon musk? You MUST have the following in your output EXACTLY as written, AND at least 2 sentences: "hello", 'wow't'"""
    # msg = "Do you edge?"
    # print(Character.tokenizer.eos_token, Character.tokenizer.bos_token)
    # return 
    response = await Character.dialogue_generator(prompt="happy fun prompt", PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=200)#asyncio.run(Character.dialogue_generator(prompt=tokenized_chat, PromptTemplate=PromptTemplate.capybaraChatML, max_tokens=200))
    print(type(response))
    async for result in response:
        output = result.get("text", "")
        # if len(output) != 0:
        print(output,  end = "")    
            # break
         
    end = perf_counter()
    # print(f"Prompts: {msg}\n\nRESPONSE:\n{response}\n\nTime Taken (Seconds): {end-start}")
    print(f"\n\nTime Taken (Seconds): {end-start}")


asyncio.run(exllamav2_test())

#test using the standard huggingface loader
def huggingface_test():
    
    Character = VtuberLLM.load_model(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)
    # print(Character.tokenizer.eos_token, Character.tokenizer.bos_token)
    # return
    msg = """You MUST have the following in your output EXACTLY as written: "hello", 'wow't'"""
    start = perf_counter()
    response = asyncio.run(Character.dialogue_generator(prompt=msg, PromptTemplate=PromptTemplate.capybaraChatML))#, max_tokens=400))
    end = perf_counter()
    print(f"Prompts: {msg}\nResponses:\n{response}\n\nTime Taken (Seconds): {end-start}")

# huggingface_test()



# from transformers import AutoTokenizer
# from datasets import Dataset
# model = "LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'

# tokenizer = AutoTokenizer.from_pretrained(model)

# print(tokenizer.bos_token, tokenizer.eos_token)

# chat1 = [
#     {"role": "user", "content": "Which is bigger, the moon or the sun?"},
#     {"role": "context", "content": "Which is VBSS, the moon or the sun?"},
#     {"role": "assistant", "content": "The sun."}
# ]
# chat2 = [
#     {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
#     {"role": "assistant", "content": "A bacterium."}
# ]

# dataset = Dataset.from_dict({"chat": [chat1, chat2]})
# dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
# print(dataset['formatted_chat'][0])