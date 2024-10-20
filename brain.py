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
import os
from dotenv import load_dotenv
from collections import deque
from multiprocessing import Queue as MPQueue
import multiprocessing

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

#retrieves live chat messages and puts them into a queue that is accessible across processes
def live_chat_process(mp_queue):
    live_chat_controller = LiveChatController.create()#handles everything regarding managing different live chats
    
    #user has not defined a livechat to attempt to retrieve livechat messages from
    if not live_chat_controller:
        mp_queue.put(None)  # Signal that no live chat is available
        logging.info("Live chat functionality is not available")
        return

    #fetch a live chat message and put it into the process queue
    async def fetch_and_send():
        while True:
            try:
                live_chat_msg = await asyncio.wait_for(live_chat_controller.fetch_chat_message(), timeout=10)
                if live_chat_msg:
                    mp_queue.put(f"{live_chat_msg[0]}: {live_chat_msg[1]}")
            except asyncio.TimeoutError:
                # No message received in 10 seconds, continue loop
                pass
            except Exception as e:
                logging.error(f"Error in live chat process: {e}")
                break
            await asyncio.sleep(15) #sleep to avoid sending too many requests for live chat messages
    
    asyncio.run(fetch_and_send())

#retrieves live chat messages from sub-process queue and puts them into a main process queue
async def live_chat_worker(mp_queue):
    while True:
        try:
            message = mp_queue.get_nowait()
            if message is None:
                logging.info("Live chat functionality was not turned on due to ENV vars being missing or set to False")
                break  # No live chat available, exit the worker
            if not live_chat_queue.full():
                await live_chat_queue.put(message)
        except multiprocessing.queues.Empty:
            # No message available, don't block
            pass
        except Exception as e:
            logging.error(f"Unexpected error in live_chat_worker: {e}")
        
        await asyncio.sleep(0.1)  # Short sleep to prevent tight looping

    logging.info("Live chat worker is exiting")


#handles stt/livechat message -> LLM output message
async def dialogue_worker():
    while True:
        try:
            #try to get STT output or live chat message
            try:
                message = speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                try:
                    message = live_chat_queue.get_nowait()
                except asyncio.QueueEmpty:
                    # Both queues are empty, wait a bit and try again
                    await asyncio.sleep(0.1)
                    continue

            print("CHOSEN MESSAGE:", message)
            chat_history = "\n".join(naive_short_term_memory)
            prompt = f"""
            Here's the previous chat history, it may or may not be relevant to the current prompt:
            {chat_history}

            The following is the current prompt:
            {message}
            """
            # print("The PROMPT:", prompt)

            #queue prevents unnecessary generation, and determines how much backlog can be generated
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
        with change_dir('./voiceAI/GPT_Test'):
            await send_tts_request(output)
        await asyncio.sleep(0.1)

#run and switch between different tasks conveniently and avoid wasting computational resources
async def loop_function():
    # Create a multiprocessing Queue for communication between processes
    mp_queue = MPQueue()
    
    # Start the live chat process
    live_chat_proc = multiprocessing.Process(target=live_chat_process, args=(mp_queue,))
    live_chat_proc.start()
    
    tasks = [
        asyncio.create_task(stt_worker()),
        asyncio.create_task(dialogue_worker()),
        asyncio.create_task(tts_worker()),
        asyncio.create_task(live_chat_worker(mp_queue))
    ]
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        # Ensure the live chat process is terminated
        live_chat_proc.terminate()
        live_chat_proc.join()


#called by run.py
if __name__ == "__main__":
    load_dotenv()#get .env file variables
    logging.basicConfig(filename="program.log", level=logging.INFO)#define log file and its level

    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)

    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    # custom_model = "LLM/unnamedSICUACCT"
    # Character = VtuberLLM.load_model(custom_model=custom_model, character_name=character_name)
    
    #exllamav2 model setup
    Character = VtuberExllamav2.load_model_exllamav2(character_name=character_name)#(generator, gen_settings, tokenizer, character_name)

    #different queues to hold messages/outputs as they finish/come for further processing
    speech_queue = asyncio.Queue(maxsize=1)
    live_chat_queue = asyncio.Queue(maxsize=1)
    llm_output_queue = asyncio.Queue(maxsize=1)

    naive_short_term_memory = deque(maxlen=6)#used to keep a very short-term memory of what was said in the current conversation
    speaker_name = "_"#temporarily emprty until finetuning has been solved fully
    
    #saves user/livechat/LLM response message data to csv file if file path is defined 
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