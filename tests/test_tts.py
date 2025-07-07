import pytest
import asyncio
from voiceAI.GPT_Test.tts_exp import send_tts_request,run_playback_thread, tts_queue
from general_utils import change_dir
import os
from unittest.mock import MagicMock

#handles turning and playing generated TTS audio from LLM's generated text
@pytest.mark.asyncio
async def test_tts():
    dummy_output = "I'm John The Magnificent, tremble before my mighty power!"
    _ = run_playback_thread(run=False)#unit test has no need for audio playback
    with change_dir('./voiceAI/GPT_Test'):
        await send_tts_request(text=dummy_output)#asyncio.wait_for(send_tts_request(dummy_output), timeout=35)
    tts_audio_bytes = tts_queue.get(timeout=1)
    print(tts_queue, tts_queue.qsize(), type(tts_audio_bytes))

    #tts queue size should be greater than 0 and type of data inside of queue should be bytes (audio stream bytes)
    assert tts_queue.qsize() > 0 and isinstance(tts_audio_bytes, bytes)