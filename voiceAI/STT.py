import speech_recognition as sr

async def speech_to_text(callback):
    """
    Speech-To-Text (STT) -- Listens to microphone and turns recognized speech to text
    
    WIP -- not the greatest recognition and very slow
    """

    #using whisper (most likely whisper small), recognizes speech and calls the callback function     
    async def recognize_speech():
        recognizer = sr.Recognizer()
        text = ""
        with sr.Microphone() as source:
            print("Speak something...")
            recognizer.energy_threshold = 2800
            audio_data = recognizer.listen(source, phrase_time_limit=15)
        try:
            text = recognizer.recognize_whisper(audio_data, language='english')
            print("You said:", text)
            await callback(text)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Error: Could not request results from Whisper service;")
        except Exception as e:
            print(f"Unexpected error at STT: {e}")

    await recognize_speech()

# #only for testing purposes
# if __name__ == "__main__":
#     import asyncio

#     #a basic callback that exists solely for basic testing of the STT
#     async def _stt_test_callback(speech):
#         #current STT system recognizes no sound as 'Thank you.' for reasons unknown
#         if speech and speech.strip() != "Thank you.":
#             await test_speech_queue.put(speech.strip())
#         print(list(test_speech_queue._queue))
    
#     test_speech_queue = asyncio.Queue(maxsize=2)
#     stt = asyncio.run(speech_to_text(_stt_test_callback))