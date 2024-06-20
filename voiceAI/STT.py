import threading
import speech_recognition as sr
def STT(callback):
    def recognize_speech():
        recognizer = sr.Recognizer()
        text = ""
        with sr.Microphone() as source:
            print("Speak something...")
            recognizer.energy_threshold = 2800
            #print("Listening...")
            audio_data = recognizer.listen(source, phrase_time_limit=27)
            #print("Processing...")

        try:
            text = recognizer.recognize_whisper(audio_data, language='english')
            print("You said:", text)
            callback(text)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: Could not request results from Whisper service;")
        except Exception as e:
            print(f"Unexpected error: {e}")

    recognize_speech()

class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()#recognizer

    def recognize_speech(self):
        with sr.Microphone() as source:
            audio_data = self.recognizer.listen(source, phrase_time_limit=5)
        text = ""
        try:
            text = self.recognizer.recognize_whisper(audio_data, language='english')
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: Could not request results from Whisper service;")
        return text

    #threading.Thread(target=recognize_speech, daemon=True).start()

if __name__ == "__main__":
    stt = SpeechToText()
    t = stt.recognize_speech()
    # def print_callback(text):
    #     print("Callback:", text)
    
    # STT(print_callback)
