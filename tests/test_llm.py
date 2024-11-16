import pytest
import asyncio
from brain import stt_worker, live_chat_process, live_chat_worker, dialogue_worker, tts_worker
from voiceAI.STT import speech_to_text

from LLM.models import VtuberExllamav2, VtuberLLM

from LLM.model_utils import LLMUtils
from LLM.llm_templates import PromptTemplate as pt
from time import perf_counter
import asyncio

# @pytest.fixture
# def speak_message():
#     print("Speak something into the microphone. The test will wait for 10 seconds.")

# #simulates stt_worker in brain.py
# @pytest.mark.asyncio
# @pytest.mark.integration
# async def test_stt_worker(speak_message):
#     #a basic callback that exists solely for basic testing of the STT
#     async def _stt_test_callback(speech):
#         #returned value from speech recognition should be of type str
#         assert isinstance(speech, str)
#         #current STT system recognizes no sound as 'Thank you.' for reasons unknown
#         if speech and speech.strip() != "Thank you.":
#             await test_speech_queue.put(speech.strip())
#         print(list(test_speech_queue._queue), test_speech_queue.full())
    
#     test_speech_queue = asyncio.Queue(maxsize=2)
#     try:
#         await asyncio.wait_for(speech_to_text(_stt_test_callback), timeout=35)
#     except Exception as e:
#         pass
#     assert not test_speech_queue.empty()

#initialize character from file 
@pytest.fixture
def character_info():
    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)
    return instructions, user_name, character_name

#define prompt template and presumed bos and eos tokens -- currenlty capybara chatml
@pytest.fixture
def prompt_template(character_info):
    instructions, user_name, character_name = character_info
    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    bos_token, eos_token = "<|im_start|>", "<|im_end|>"

    return PromptTemplate.capybaraChatML, bos_token, eos_token
 
#test that character information can be loaded in
def test_character_loading(character_info):
    #get current character's information to use
    instructions, user_name, character_name = character_info
    assert instructions and user_name and character_name
    assert isinstance(instructions, str) and isinstance(user_name, str) and isinstance(character_name, str)

#test that prompt template is correctly initialized
def test_PromptTemplate_population(prompt_template):
    PromptTemplate, bos_token, eos_token = prompt_template#pt(instructions_string, user_name, character_name)
    # print(PromptTemplate.capybaraChatML())
    capybaraChatML = PromptTemplate()
    assert isinstance(capybaraChatML, str) and bos_token in capybaraChatML and eos_token in capybaraChatML

def test_model_loading():
    model = "LLM/ALlama"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
    Character = VtuberExllamav2.load_model_exllamav2(model_dir=model,character_name="John")
    print(Character, type(Character))

#test using exllamav2
def test_exllamav2(prompt_template):
    model = "LLM/ALlama"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'
    prompTemplate, bos_token, eos_token = prompt_template
    Character = VtuberExllamav2.load_model_exllamav2(model_dir=model,character_name="John")#(generator, gen_settings, tokenizer, character_name)
    #msg = "<|begin_of_text|>instructions: John Smith takes everything to an extreme. He dresses in outlandish and flamboyant clothing, often with theatricality. His speech is full of bombastic, grand and dramatic pronouncements. Everytime he speaks he says and responds ALWAYS with something NEW and UNIQUE EVERY SINGLE TIME. You will be penalized for saying anything present in this paragraph, and for not saying random unique things. He speaks of many fascinating topics.\n\n user message: Kid Flasher: Elon musk? You MUST have the following in your output EXACTLY as written, AND at least 2 sentences: "hello", 'wow't'"""
    msg = "Do you edge?"
    MAX_LATENCY = 2.00 #seconds
    start = perf_counter()
    print(Character.tokenizer.eos_token, Character.tokenizer.bos_token)
    #returns generated dialogue
    response = asyncio.run(Character.dialogue_generator(prompt=msg, PromptTemplate=prompTemplate, max_tokens=200))
    end = perf_counter()

    time_taken = end-start
    # print(f"Prompts: {msg}\n\nRESPONSE:\n{response}\n\nTime Taken (Seconds): {time_taken}")
    #response should not take more than 2 seconds
    assert time_taken <= MAX_LATENCY
    #bos and eos tokens should never be in the received responses
    assert response and isinstance(response, str) and bos_token not in response and eos_token not in response

#tests random token length generator method from LLMUtils
def test_random_token_len():
    min_tokens = 15
    max_tokens = 200
    input_len = 0 #how long the input text itself is -- not used anymore possibly
    token_lengths = []#holds different to generate token numbers
    
    #min and max number of to generate should never be 0
    assert min_tokens != 0 and max_tokens != 0
    for _ in range(5000):
        num_tokens_to_generate = LLMUtils.get_rand_token_len(min_tokens, max_tokens, input_len)
        token_lengths.append(num_tokens_to_generate)

    #the number of tokens to generate should not be less than min_tokens nor greater than max tokens
    any_token_too_short = any(token_length < min_tokens for token_length in token_lengths)
    any_token_too_long = any(token_length > max_tokens for token_length in token_lengths)
    # print(any_token_too_long, any_token_too_short, max(token_lengths))
    
    assert any_token_too_short == False and any_token_too_long == False