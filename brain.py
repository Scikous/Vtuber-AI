from LLM.models import VtuberExllamav2#, VtuberLLM
from LLM.model_utils import LLMUtils
from LLM.llm_templates import PromptTemplate as pt
from livechatAPI.livechat import LiveChatController
from voiceAI.TTS import send_tts_request, tts_queue
from voiceAI.STT import speech_to_text
import logging
import asyncio
import time
from dotenv import load_dotenv


#handles speech-to-text in the background
async def stt_worker():
    async def stt_callback(speech):
        #current STT system recognizes no sound as 'Thank you.' for reasons unknown
        if speech and speech.strip() != "Thank you.":
            await speech_queue.put(speech.strip())
        print(list(speech_queue._queue))

    while True:
        await speech_to_text(stt_callback)
        await asyncio.sleep(0.1)

#handles retrieving a random chat message in the background
async def live_chat_worker(live_chat_setup):
    while True:
        live_chat_msg = await live_chat_setup.fetch_chat_message()
        if live_chat_msg:
            await live_chat_queue.put(f"{live_chat_msg[0]}: {live_chat_msg[1]}")
        await asyncio.sleep(0.1)

#handles stt/livechat message -> LLM output message
async def dialogue_worker():
    while True:
        try:
            # Check if there's a live chat message first
            if not live_chat_queue.empty():
                message = await live_chat_queue.get()
            else:
                # Otherwise, get speech from the speech queue
                message = await speech_queue.get()
                
            print("CHOSEN MESSAGE:", message)
            #avoid generating too much text for the TTS to speak outloud
            if not tts_queue.full():
                output = await Character.dialogue_generator(message, PromptTemplate.capybaraChatML, max_tokens=100)
                await output_queue.put(output)
            else:
                print("TTS queue is full, skipping generation.")
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error at worker: {e}")

#handles turning and playing generated TTS audio from LLM's generated text
async def tts_worker():
    while True:
        output = await output_queue.get()
        await send_tts_request(output)
        await asyncio.sleep(0.1)

#run and switch between different tasks conveniently and avoid wasting computational resources
async def loop_function(live_chat_setup):
    stt_task = asyncio.create_task(stt_worker())
    live_chat_task = asyncio.create_task(live_chat_worker(live_chat_setup))
    dialogue_task = asyncio.create_task(dialogue_worker())
    tts_task = asyncio.create_task(tts_worker())

    await asyncio.gather(stt_task, live_chat_task, dialogue_task, tts_task)

#called by run.py
if __name__ == "__main__":
    load_dotenv()#get .env file variables

    custom_model = "LLM/unnamedSICUACCT"
    model, tokenizer = LLMUtils.load_model(custom_model_name=custom_model)
    
    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    # Character = VtuberLLM(model, tokenizer, character_name)  
    
    #exllamav2 model
    # generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2() #deprecated
    Character = VtuberExllamav2.load_model_exllamav2(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)

    speech_queue = asyncio.Queue(maxsize=2)
    live_chat_queue = asyncio.Queue(maxsize=2)
    output_queue = asyncio.Queue(maxsize=2)

    fetch_twitch = True
    fetch_youtube = True
    live_chat_setup = LiveChatController(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube)

    asyncio.run(loop_function(live_chat_setup))
