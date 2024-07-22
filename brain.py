# from LLM.models import VtuberExllamav2, VtuberLLM
# from LLM.model_utils import LLMUtils
# from voiceAI.TTS import send_tts_request, tts_queue
# from LLM.llm_templates import PromptTemplate as pt   # PromptTemplate as pt
# from voiceAI.STT import STT
# import logging
# import asyncio
# import time

# from livechatAPI.livechat import LiveChatSetup


# async def stt_worker():
#     async def stt_callback(speech):
#         if speech and speech.strip() != "Thank you.":
#             await speech_queue.put(speech.strip())
#         print(list(speech_queue._queue))

#     while True:
#         await STT(stt_callback)
#         await asyncio.sleep(0.1)

# async def dialogue_worker():
#     while True:
#         try:
#             # live_chat_msg = await live_chat_setup.fetch_chat_message()
#             # if live_chat_msg:
#             #     live_chat_msg = f"{live_chat_msg[0]}: {live_chat_msg[1]}"
#             # else:
#             #     speech = await speech_queue.get()
#             # comment = live_chat_msg if live_chat_msg else speech

#             speech = await speech_queue.get()
#             comment = speech
#             print(comment)
#             if not tts_queue.full():
#                 output = await Character.dialogue_generator(comment, PromptTemplate.capybaraChatML, max_tokens=100)
#                 # await asyncio.sleep(5)
#                 # await send_tts_request(output)
#                 # print("THE OUTPUT IS", output)
#                 await output_queue.put(output)
#                 # await send_tts_request(output)
#             else:
#                 print("TTS queue is full, skipping generation.")
#         except ValueError:
#             pass
#         except Exception as e:
#             logging.error(f"Unexpected error at worker: {e}")

# async def tts_worker():
#     while True:
#         output = await output_queue.get()
#         await send_tts_request(output)
#         await asyncio.sleep(0.1)

# async def loop_function():
#     stt_task = asyncio.create_task(stt_worker())
#     dialogue_task = asyncio.create_task(dialogue_worker())
#     tts_task = asyncio.create_task(tts_worker())

#     await asyncio.gather(*[stt_task, dialogue_task, tts_task])

# if __name__ == "__main__":
#     custom_model = "LLM/unnamedSICUACCT"
#     model, tokenizer = LLMUtils.load_model(custom_model_name=custom_model)
    
#     character_info_json = "LLM/characters/character.json"
#     instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

#     instructions_string = f"""{instructions}"""
#     PromptTemplate = pt(instructions_string, user_name, character_name)
#     # Character = VtuberLLM(model, tokenizer, character_name)  
    
#     #exllamav2 model
#     generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2()
#     Character = VtuberExllamav2(generator, gen_settings, tokenizer, character_name)

#     speech_queue = asyncio.Queue(maxsize=2)
#     output_queue = asyncio.Queue(maxsize=2)

#     # fetch_twitch = True
#     # fetch_youtube = True
#     # kick = []  # Placeholder for Kick messages
#     # live_chat_setup = LiveChatSetup(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube)

#     asyncio.run(loop_function())


from LLM.models import VtuberExllamav2, VtuberLLM
from LLM.model_utils import LLMUtils
from voiceAI.TTS import send_tts_request, tts_queue
from LLM.llm_templates import PromptTemplate as pt
from voiceAI.STT import STT
import logging
import asyncio
import time

from livechatAPI.livechat import LiveChatSetup

async def stt_worker():
    async def stt_callback(speech):
        if speech and speech.strip() != "Thank you.":
            await speech_queue.put(speech.strip())
        print(list(speech_queue._queue))

    while True:
        await STT(stt_callback)
        await asyncio.sleep(0.1)

async def live_chat_worker(live_chat_setup):
    while True:
        live_chat_msg = await live_chat_setup.fetch_chat_message()
        if live_chat_msg:
            await live_chat_queue.put(f"{live_chat_msg[0]}: {live_chat_msg[1]}")
        await asyncio.sleep(0.1)

async def dialogue_worker():
    while True:
        try:
            # Check if there's a live chat message first
            if not live_chat_queue.empty():
                comment = await live_chat_queue.get()
            else:
                # Otherwise, get speech from the speech queue
                comment = await speech_queue.get()
                
            print("CHOSEN MESSAGE:", comment)
            if not tts_queue.full():
                output = await Character.dialogue_generator(comment, PromptTemplate.capybaraChatML, max_tokens=100)
                await output_queue.put(output)
            else:
                print("TTS queue is full, skipping generation.")
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error at worker: {e}")

async def tts_worker():
    while True:
        output = await output_queue.get()
        await send_tts_request(output)
        await asyncio.sleep(0.1)

async def loop_function(live_chat_setup):
    stt_task = asyncio.create_task(stt_worker())
    live_chat_task = asyncio.create_task(live_chat_worker(live_chat_setup))
    dialogue_task = asyncio.create_task(dialogue_worker())
    tts_task = asyncio.create_task(tts_worker())

    await asyncio.gather(stt_task, live_chat_task, dialogue_task, tts_task)

if __name__ == "__main__":
    custom_model = "LLM/unnamedSICUACCT"
    model, tokenizer = LLMUtils.load_model(custom_model_name=custom_model)
    
    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    # Character = VtuberLLM(model, tokenizer, character_name)  
    
    #exllamav2 model
    generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2()
    Character = VtuberExllamav2(generator, gen_settings, tokenizer, character_name)

    speech_queue = asyncio.Queue(maxsize=2)
    live_chat_queue = asyncio.Queue(maxsize=2)
    output_queue = asyncio.Queue(maxsize=2)

    fetch_twitch = True
    fetch_youtube = True
    live_chat_setup = LiveChatSetup(fetch_twitch=fetch_twitch, fetch_youtube=fetch_youtube)

    asyncio.run(loop_function(live_chat_setup))
