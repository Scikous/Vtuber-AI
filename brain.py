from utils import model_loader, dialogue_generator, character_reply_cleaner
from TTS import send_tts_request
import prompt_templates as pt
from STT import STT
import time
from queue import Queue
import logging
from tkinter import Tk, Button
import threading
# def stt_worker():
#   """
#   This function runs in the background and performs speech recognition.

#   It continuously calls `STT()` and puts the recognized speech text into the queue.
#   """
#   while True:
#     speech = STT()
#     speech_queue.put(speech)

# def loop_function():
#   # Start the STT worker thread
#   stt_thread = threading.Thread(target=stt_worker, daemon=True)
#   stt_thread.start()
#   print("hllelo")

#   while True:
#       # Get the next recognized speech from the queue (waits if empty)
#       if(not speech_queue.empty()):
#         logging.debug(f"Speech queue: {speech_queue.queue}")
#         speech = speech_queue.get(timeout=1)  # Set a timeout to avoid waiting indefinitely
#         comments = [speech]
#         outputs = dialogue_generator(model, tokenizer, comments, prompt_template.capybaraChatML)
#         clean_reply = character_reply_cleaner(outputs[0]).lower()
#         send_tts_request(clean_reply)
#       else:
#         pass

# if __name__ == "__main__":
#   custom_model = "unnamedSICUACCT"
#   model, tokenizer = model_loader(custom_model_name=custom_model)

#   with open("characters/character.txt", "r") as f:
#       character_info = f.readline()
#   instructions_string = f"""{character_info}"""
#   #print(instructions_string)
#   prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''
#   #comments = ["How are you feeling today?"]
#   prompt_template = pt.prompt_template(instructions_str=instructions_string, character_name="John")

#   # Define the queue to store recognized speech text
#   speech_queue = Queue()

#           # ... (your existing processing logic with 'outputs')
#   loop_function()


  #loop_thread = threading.Thread(target=loop_function)
  #loop_thread.start()  # Start the loop in a separate thread

  #root.mainloop()  # Keeps the window open and waits for interaction

  #print(outputs)



def stt_worker():
    def stt_callback(speech):
        speech_queue.put(speech)
        print(speech_queue.queue)


    while True:
        STT(stt_callback)
        # Add a short sleep to avoid tight loop
        time.sleep(1)

def loop_function():
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    while True:
        try:
            speech = speech_queue.get(timeout=1)
            #print(speech_queue.queue)
            #logging.debug(f"Speech queue: {speech_queue.queue}")
            comments = [speech]
            outputs = dialogue_generator(model, tokenizer, comments, prompt_template.capybaraChatML)
            clean_reply = character_reply_cleaner(outputs[0]).lower()
            send_tts_request(clean_reply)
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    custom_model = "unnamedSICUACCT"
    model, tokenizer = model_loader(custom_model_name=custom_model)

    with open("characters/character.txt", "r") as f:
        character_info = f.readline()
    instructions_string = f"""{character_info}"""
    prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''
    prompt_template = pt.prompt_template(instructions_str=instructions_string, character_name="John")

    speech_queue = Queue()

    loop_function()
