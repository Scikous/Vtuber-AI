# import speech_recognition as sr

# async def speech_to_text(callback):
#     """
#     Speech-To-Text (STT) -- Listens to microphone and turns recognized speech to text
    
#     WIP -- not the greatest recognition and very slow
#     """

#     #using whisper (most likely whisper small), recognizes speech and calls the callback function     
#     async def recognize_speech():
#         recognizer = sr.Recognizer()
#         text = ""
#         with sr.Microphone() as source:
#             print("Speak something...")
#             recognizer.energy_threshold = 2800
#             audio_data = recognizer.listen(source, phrase_time_limit=15)
#         try:
#             text = recognizer.recognize_whisper(audio_data, language='english')
#             print("You said:", text)
#             await callback(text)
#         except sr.UnknownValueError:
#             print("Sorry, could not understand audio.")
#         except sr.RequestError as e:
#             print("Error: Could not request results from Whisper service;")
#         except Exception as e:
#             print(f"Unexpected error at STT: {e}")

#     await recognize_speech()

import speech_recognition as sr
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=1)

async def speech_to_text(callback):
    """
    Speech-To-Text (STT) -- Listens to microphone and turns recognized speech to text
    
    Non-blocking implementation using asyncio and ThreadPoolExecutor
    """
    
    def recognize_speech_sync():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak something...")
            recognizer.energy_threshold = 2800
            try:
                audio_data = recognizer.listen(source, timeout=1, phrase_time_limit=15)
                text = recognizer.recognize_whisper(audio_data, language='english')
                print("You said:", text)
                return text
            except sr.WaitTimeoutError:
                return None
            except sr.UnknownValueError:
                print("Sorry, could not understand audio.")
                return None
            except sr.RequestError as e:
                print("Error: Could not request results from Whisper service;")
                return None
            except Exception as e:
                print(f"Unexpected error at STT: {e}")
                return None

    while True:
        try:
            # Run the speech recognition in a separate thread
            text = await asyncio.get_event_loop().run_in_executor(executor, recognize_speech_sync)
            
            if text:
                await callback(text)
            
            # Small delay to prevent busy-waiting
            await asyncio.sleep(0.1)
        
        except Exception as e:
            print(f"Unexpected error in speech_to_text: {e}")
            await asyncio.sleep(1)  # Wait a bit before retrying

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