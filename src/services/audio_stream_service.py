import pyaudio
import wave
import asyncio
from typing import Optional
from .base_service import BaseService

class AudioStreamService(BaseService):
    def __init__(self, shared_resources=None):
        super().__init__(shared_resources)
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 32000
        self.chunk_size = 1024

    def _open_stream(self):
        if not self.stream or not self.stream.is_active():
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )

    async def run_worker(self):
        self._open_stream()
        audio_queue = self.shared_resources['queues']['audio_output_queue']
        prefetch_buffer = asyncio.Queue(maxsize=5)  # Buffer 5 chunks ahead

        async def prefetch_task():
            while True:
                chunk = await audio_queue.get()
                await prefetch_buffer.put(chunk)

        asyncio.create_task(prefetch_task())

        while True:
            try:
                # Get next chunk while current plays
                wav_bytes = await prefetch_buffer.get()
                self.stream.write(wav_bytes[64:])
                audio_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audio stream error: {e}")
                self._open_stream()

    def __del__(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()