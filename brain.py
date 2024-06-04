from LLM.utils import model_loader, dialogue_generator, character_reply_cleaner
from voiceAI.TTS import send_tts_request
import LLM.prompt_templates as pt
from voiceAI.STT import STT
import time
from queue import Queue
import logging
import threading

def stt_worker():
    def stt_callback(speech):
        #keep conversation on the most recent topic - ex. too long of a queue = model is responding to something 5 minutes ago and not current topic
        if speech_queue.full():
            speech_queue.get()
            speech_queue.get()
            speech_queue.get()
            #speech_queue.queue.clear()
        speech_queue.put(speech)
        print(speech_queue.queue)


    while True:
        STT(stt_callback)
        # Add a short sleep to avoid tight loop
        time.sleep(5)

def loop_function():
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    while True:
        try:
            speech = speech_queue.get(timeout=1)
            #print(speech_queue.queue)
            #logging.debug(f"Speech queue: {speech_queue.queue}")
            comment = speech
            output = dialogue_generator(model, tokenizer, comment, prompt_template.capybaraChatML)
            print("THEOUTPUT IS", output)
            #clean_reply = character_reply_cleaner(output).lower()
            send_tts_request(output)
            time.sleep(7)
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    custom_model = "LLM/unnamedSICUACCT"
    model, tokenizer = model_loader(custom_model_name=custom_model)

    with open("characters/character.txt", "r") as f:
        character_info = f.readline()
    instructions_string = f"""{character_info}"""
    prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''
    prompt_template = pt.prompt_template(instructions_str=instructions_string, character_name="John")

    speech_queue = Queue(maxsize=5)

    loop_function()
