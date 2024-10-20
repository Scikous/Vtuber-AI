import pytest
import asyncio
from brain import stt_worker, live_chat_process, live_chat_worker, dialogue_worker, tts_worker
from voiceAI.STT import speech_to_text






@pytest.fixture
def speak_message():
    print("Speak something into the microphone. The test will wait for 10 seconds.")

#simulates stt_worker in brain.py
@pytest.mark.asyncio
async def test_stt_worker(speak_message):
    #a basic callback that exists solely for basic testing of the STT
    async def _stt_test_callback(speech):
        #current STT system recognizes no sound as 'Thank you.' for reasons unknown
        if speech and speech.strip() != "Thank you.":
            await test_speech_queue.put(speech.strip())
        print(list(test_speech_queue._queue), test_speech_queue.full())
    
    test_speech_queue = asyncio.Queue(maxsize=2)
    try:
        await asyncio.wait_for(speech_to_text(_stt_test_callback), timeout=35)
    except Exception as e:
        pass
    assert not test_speech_queue.empty()
