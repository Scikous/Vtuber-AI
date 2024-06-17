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
            try:
                audio_data = recognizer.listen(source, timeout=5)
            except Exception as e:
                print("har har har", e)
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

    #threading.Thread(target=recognize_speech, daemon=True).start()

if __name__ == "__main__":
    def print_callback(text):
        print("Callback:", text)
    
    STT(print_callback)
