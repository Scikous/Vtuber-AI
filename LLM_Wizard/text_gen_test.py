from models import VtuberExllamav2, VtuberLLM
from huggingface_hub import snapshot_download
from model_utils import LLMUtils
from time import perf_counter
import asyncio

# from general_utils import read_messages_csv

#get current character's information to use
character_info_json = "LLM_Wizard/characters/character.json"
instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

#set prompt template to follow current character's instruction set and name
instructions_string = f"""{instructions}"""
dummy_data = ["Good day, state your name.", "What is your favorite drink?", "Do you edge?"]

#LLM model to use
model = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
# model = snapshot_download(repo_id=model_name)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model) # Example

# print(f"Model '{model_name}' is located locally at: {model}")

#test using the exllamav2
async def exllamav2_test():
    Character = VtuberExllamav2.load_model(model=model,character_name=character_name)#(generator, gen_settings, tokenizer, character_name)

    start = perf_counter()
    prompt = LLMUtils.prompt_wrapper("Happy fun prompt!", "User is happy to talk with you")
    response = await Character.dialogue_generator(prompt=prompt, max_tokens=200)
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
    response = asyncio.run(Character.dialogue_generator(prompt=msg))#, max_tokens=400))
    end = perf_counter()
    print(f"Prompts: {msg}\nResponses:\n{response}\n\nTime Taken (Seconds): {end-start}")
