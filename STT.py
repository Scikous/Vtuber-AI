import speech_recognition as sr
import time

def STT():
    # Create a Recognizer instance
    recognizer = sr.Recognizer()
    # Capture audio input from the microphone
    while True:
        with sr.Microphone() as source:
            print("Speak something...")
            audio_data = recognizer.listen(source)

        # Perform speech recognition using Google Web Speech API
        try:
            text = recognizer.recognize_google(audio_data)
            print("You said:", text)
            break
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: Could not request results from Google Speech Recognition service;")
    return text
    