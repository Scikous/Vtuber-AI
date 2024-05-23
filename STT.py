# import speech_recognition as sr

# import time



# def STT():
#     # Create a Recognizer instance
#     recognizer = sr.Recognizer()
#     text = ""
#     # Capture audio input from the microphone
#     with sr.Microphone() as source:
#         print("Speak something...")
#         recognizer.energy_threshold = 2800
#         print("hello")
#         audio_data = recognizer.listen(source, timeout=5)
#         print("Whau")
#     # Perform speech recognition using Google Web Speech API
#     try:
#         text = recognizer.recognize_whisper(audio_data, language='english')
#         print("You said:", text)
#     except sr.UnknownValueError:
#         print("Sorry, could not understand audio.")
#     except sr.RequestError as e:
#         print("Error: Could not request results from Whisper service;")
#     return text
    

# if __name__ == "__main__":
#     STT()



import threading
import speech_recognition as sr

def STT(callback):
    def recognize_speech():
        recognizer = sr.Recognizer()
        text = ""
        with sr.Microphone() as source:
            print("Speak something...")
            recognizer.energy_threshold = 2200
            #print("Listening...")
            audio_data = recognizer.listen(source, timeout=5)
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

    #threading.Thread(target=recognize_speech).start()

if __name__ == "__main__":
    def print_callback(text):
        print("Callback:", text)
    
    STT(print_callback)
