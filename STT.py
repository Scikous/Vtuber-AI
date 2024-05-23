import speech_recognition as sr

import time

def STT():
    # Create a Recognizer instance
    recognizer = sr.Recognizer()
    text = ""
    # Capture audio input from the microphone
    with sr.Microphone() as source:
        print("Speak something...")
        recognizer.energy_threshold = 1800
        audio_data = recognizer.listen(source)
    # Perform speech recognition using Google Web Speech API
    try:
        text = recognizer.recognize_whisper(audio_data, language='english')
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
    except sr.RequestError as e:
        print("Error: Could not request results from Whisper service;")
    return text
    

if __name__ == "__main__":
    STT()
