from utils import model_loader, dialogue_generator, character_reply_cleaner
from TTS import send_tts_request
import prompt_templates as pt
from STT import STT
import time

from tkinter import Tk, Button
import threading
if __name__ == "__main__":
  custom_model = "unnamedSICUACCT"
  model, tokenizer = model_loader(custom_model_name=custom_model)

  with open("characters/character.txt", "r") as f:
      character_info = f.readline()
  instructions_string = f"""{character_info}"""
  #print(instructions_string)
  prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''
  #comments = ["How are you feeling today?"]
  prompt_template = pt.prompt_template(instructions_str=instructions_string, character_name="John")

  running = True  # Flag to control loop state

  def toggle_loop():
    global running
    running = not running  # Flips the state (True/False)
    print("stopped", running)

  def loop_function():
    # Your code that runs indefinitely here
    # (e.g., print("Looping..."), perform calculations, etc.)
    while True:
      if running:
          speech = STT()
          #print(speech)
          comments = [speech]
          outputs = dialogue_generator(model, tokenizer, comments, prompt_template.capybaraChatML)
          #print("#"*30, "\n",outputs[0], "\n", "#"*30)
          clean_reply = character_reply_cleaner(outputs[0]).lower()
          #print("#"*30, "\n",clean_reply, "\n", "#"*30)
          send_tts_request(clean_reply)
          # Add a short delay to avoid excessive updates (optional)
      else:
        pass
      #time.sleep(5)

  loop_function()
  #loop_thread = threading.Thread(target=loop_function)
  #loop_thread.start()  # Start the loop in a separate thread

  #root.mainloop()  # Keeps the window open and waits for interaction

  #print(outputs)


