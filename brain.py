from LLM.models import VtuberExllamav2, VtuberLLM
from LLM.model_utils import LLMUtils
from voiceAI.TTS import send_tts_request
from LLM.llm_templates import PromptTemplate as pt   # PromptTemplate as pt
from voiceAI.STT import STT, SpeechToText
import time
from queue import Queue
import logging
import threading


def stt_worker():
    def stt_callback(speech):
        # keep conversation on the most recent topic - ex. too long of a queue = model is responding to something 5 minutes ago and not current topic
        if speech_queue.full():
            speech_queue.get()
            speech_queue.get()
            speech_queue.get()
            # speech_queue.queue.clear()
        if speech and speech.strip() != "Thank you.":
            speech_queue.put(speech.strip())
        print(speech_queue.queue)

    while True:
        STT(stt_callback)
        # Add a short sleep to avoid tight loop
        time.sleep(2)


def loop_function():
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    while True:
        try:
            speech = speech_queue.get(timeout=1)
            # print(speech_queue.queue)
            # logging.debug(f"Speech queue: {speech_queue.queue}")
            comment = speech
            output = Character.dialogue_generator(comment, PromptTemplate.capybaraChatML)
            print("THEOUTPUT IS", output)
            # clean_reply = character_reply_cleaner(output).lower()
            send_tts_request(output)
            # time.sleep(7)
        except ValueError:
            pass
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

######################################################################################################################
#WIP
# from threading import Thread

# class STTWorker(Thread):
#     def __init__(self, speech_queue):
#         Thread.__init__(self)
#         # self.recognizer = recognizer
#         self.speech_queue = speech_queue
#         # self.daemon = True


#     def run(self):
#         stt = SpeechToText()
#         while True:
#             print("hello from STT")
#             speech = stt.recognize_speech()
#             print("hello", speech)
#             self.speech_queue.put(speech)
#             print(f"STTWorker recognized: {speech}")
#             time.sleep(0.1)

# class TTSWorker(Thread):
#     def __init__(self, tts_queue):
#         Thread.__init__(self)
#         # self.tts_engine = tts_engine
#         self.tts_queue = tts_queue
#         self.daemon = True



#     def run(self):
#         # tts = TextToSpeech(self.tts_engine)
#         print("hello from TTS")
#         while True:
#             if not self.tts_queue.empty():
#                 text_to_speak = self.tts_queue.get()
#                 send_tts_request(text_to_speak)
#                 # tts.speak_text(text_to_speak)
#                 print(f"TTSWorker spoke: {text_to_speak}")
#             time.sleep(0.1)

# class DialogueWorker(Thread):
#     def __init__(self, vtuber_llm, speech_queue, tts_queue, prompt_template):
#         Thread.__init__(self)
#         # self.daemon = True
#         self.vtuber_llm = vtuber_llm
#         self.speech_queue = speech_queue
#         self.tts_queue = tts_queue
#         self.prompt_template = prompt_template


#     def run(self):
#         print("hello from dialogue")
#         while True:
#             if not self.speech_queue.empty():
#                 comment = self.speech_queue.get()
#                 response = self.vtuber_llm.dialogue_generator(comment, self.prompt_template)
#                 self.tts_queue.put(response)
#                 print(f"DialogueWorker generated response: {response}")
#             time.sleep(0.1)

######################################################################################################################

if __name__ == "__main__":
    custom_model = "LLM/unnamedSICUACCT"
    model, tokenizer = LLMUtils.load_model(custom_model_name=custom_model)
    
    character_info_json = "LLM/characters/character.json"
    instructions, user_name, character_name = LLMUtils.load_character(character_info_json)
    
    # with open("LLM/characters/character.txt", "r") as f:
    #     character_info = f.readline()
    instructions_string = f"""{instructions}"""
    PromptTemplate = pt(instructions_string, user_name, character_name)
    # Character = VtuberLLM(model, tokenizer, character_name)  
    
    #exllamav2 model
    generator, gen_settings, tokenizer = LLMUtils.load_model_exllamav2()
    Character = VtuberExllamav2(generator, gen_settings, tokenizer, character_name)

    speech_queue = Queue(maxsize=5)
    tts_queue = Queue()

    # stt_worker = STTWorker(speech_queue)
    # dialogue_worker = DialogueWorker(Character, speech_queue, tts_queue, PromptTemplate.capybaraChatML)
    # tts_worker = TTSWorker(tts_queue)

    # stt_worker.start()
    # dialogue_worker.start()
    # tts_worker.start()

    # stt_worker.join()
    # dialogue_worker.join()
    # tts_worker.join()

    loop_function()