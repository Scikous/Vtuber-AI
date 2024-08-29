from LLM.models import VtuberExllamav2#, VtuberLLM
from LLM.model_utils import LLMUtils
from LLM.llm_templates import PromptTemplate as pt
from livechatAPI.livechat import LiveChatController
from general_utils import get_env_var, write_messages_csv, change_dir
# from voiceAI.TTS import send_tts_request, tts_queue
from voiceAI.GPT_Test.tts_exp import send_tts_request, tts_queue
from voiceAI.STT import speech_to_text
import logging
import asyncio
import time, os
from dotenv import load_dotenv
from collections import deque


#handles speech-to-text in the background
async def stt_worker():
    async def stt_callback(speech):
        #current STT system recognizes no sound as 'Thank you.' for reasons unknown
        if speech and speech.strip() != "Thank you.":
            await speech_queue.put(f"{speaker_name}: {speech.strip()}")
        print(list(speech_queue._queue))

    while True:
        await speech_to_text(stt_callback)
        await asyncio.sleep(0.1)

#handles retrieving a random chat message in the background
async def live_chat_worker(live_chat_controller):
    while True:
        if not live_chat_queue.full():
            live_chat_msg = await live_chat_controller.fetch_chat_message()
            if live_chat_msg:
                await live_chat_queue.put(f"{live_chat_msg[0]}: {live_chat_msg[1]}")
        await asyncio.sleep(50)

#handles stt/livechat message -> LLM output message
async def dialogue_worker():
    while True:
        try:
            # # Check if there's a live chat message first
            # if not live_chat_queue.empty():
            #     message = await live_chat_queue.get()
            # else:
            #     # Otherwise, get speech from the speech queue
            #     message = await speech_queue.get()
                
            # Create tasks for getting messages from both queues
            speech_task = asyncio.create_task(speech_queue.get())
            live_chat_task = asyncio.create_task(live_chat_queue.get())

            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [speech_task, live_chat_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the pending task
            for task in pending:
                task.cancel()

            # Get the completed task's result
            completed_task = done.pop()
            message = await completed_task

            print("CHOSEN MESSAGE:", message)
            chat_history = "\n".join(naive_short_term_memory)
            prompt = f"""
            Here's the previous chat history, it may or may not be relevant to the current prompt:
            {chat_history}

            The following is the current prompt:
            {message}
            """
            print("The PROMPT:", prompt)


            #avoid generating too much text for the TTS to speak outloud
            if not tts_queue.full():
                output = await Character.dialogue_generator(prompt, PromptTemplate.capybaraChatML, max_tokens=100)
                await llm_output_queue.put(output)

                naive_short_term_memory.append(message)
                naive_short_term_memory.append(f"{character_name}: {output}")
                #write message to file -- stability is questionable for bigger stream chats
                if write_message:
                   await write_message(conversation_log_file, message_data=(message, output))
            else:
                print("TTS queue is full, skipping generation.")
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error at worker: {e}")

#handles turning and playing generated TTS audio from LLM's generated text
async def tts_worker():
    while True:
        output = await llm_output_queue.get()
        import os
        with change_dir('./voiceAI/GPT_Test'):
            print(os.getcwd())
            await send_tts_request(output)
        await asyncio.sleep(0.1)

#run and switch between different tasks conveniently and avoid wasting computational resources
async def loop_function():
    tasks = [
        asyncio.create_task(stt_worker()),
        asyncio.create_task(dialogue_worker()),
        asyncio.create_task(tts_worker())
    ]
    

    live_chat_controller = LiveChatController.create()
    #it is possible that no livechats are being used
    if live_chat_controller:
        tasks.append(asyncio.create_task(live_chat_worker(live_chat_controller)))
    
    await asyncio.gather(*tasks)


#called by run.py
if __name__ == "__main__":
    load_dotenv()#get .env file variables

    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    # custom_model = "LLM/unnamedSICUACCT"
    # Character = VtuberLLM.load_model(custom_model=custom_model, character_name=character_name)
    
    #exllamav2 model
    # generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2() #deprecated
    Character = VtuberExllamav2.load_model_exllamav2(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)

    speech_queue = asyncio.Queue(maxsize=1)
    live_chat_queue = asyncio.Queue(maxsize=1)
    llm_output_queue = asyncio.Queue(maxsize=1)

    naive_short_term_memory = deque(maxlen=6)
    speaker_name = "_"#temporary until finetuning has been solved fully
    #saves user/livechat, LLM response message data if file path is provided 
    conversation_log_file=get_env_var("CONVERSATION_LOG_FILE")
    if conversation_log_file:
        #check if path is full path (absolute path or not) -- add root directory extension if not abs path
        if not os.path.isabs(conversation_log_file):
            project_root = os.path.dirname(__file__) + '/'
            conversation_log_file = project_root + conversation_log_file
        write_message = write_messages_csv #pre-assign user message + LLM response writing function

    #ENV variables determine whether to fetch specific livechats
    fetch_youtube = get_env_var("YT_FETCH") 
    fetch_twitch = get_env_var("TW_FETCH")
    fetch_kick = get_env_var("KI_FETCH")

    asyncio.run(loop_function())
