import asyncio
from typing import Optional
from .base_service import BaseService
from .audio_playback_base import AudioPlaybackBase
from .pyaudio_playback import PyAudioPlayback # Default backend

class AudioStreamService(BaseService):
    def __init__(self, shared_resources=None, audio_playback_backend: Optional[AudioPlaybackBase] = None):
        super().__init__(shared_resources)
        
        if audio_playback_backend:
            self.audio_playback_backend = audio_playback_backend
        else:
            # Default to PyAudioPlayback if no backend is provided
            # Configuration for the default backend can be passed via shared_resources or a dedicated config
            backend_config = self.shared_resources.get('config', {}).get('audio_backend_settings', {})
            self.audio_playback_backend = PyAudioPlayback(config=backend_config, logger=self.logger)
        
        self.logger.info(f"AudioStreamService initialized with backend: {type(self.audio_playback_backend).__name__}")

    def _ensure_stream_is_open(self):
        """Ensures the audio stream is open, attempting to open it if not."""
        try:
            if not self.audio_playback_backend.is_active():
                self.logger.info("Audio stream is not active. Attempting to open...")
                self.audio_playback_backend.open_stream()
        except Exception as e:
            self.logger.error(f"Failed to open or ensure audio stream is active: {e}")
            # Depending on the error, might want to re-raise or handle gracefully
            raise # Re-raise to stop the worker if stream cannot be opened

    async def _play_buffered_audio(self, playback_buffer: list):
        """Plays the content of the playback_buffer and clears it asynchronously."""
        if not playback_buffer:
            return
        try:
            self._ensure_stream_is_open()
            full_audio_data = b''.join(chunk for chunk in playback_buffer)
            if full_audio_data: # Ensure there's data to play
                loop = asyncio.get_running_loop()
                # Run the blocking write_chunk in a separate thread
                await loop.run_in_executor(None, self.audio_playback_backend.write_chunk, full_audio_data)
        except Exception as e:
            self.logger.error(f"Error playing buffered audio: {e}")
        finally:
            playback_buffer.clear()

    async def run_worker(self):
        self.logger.info("AudioStreamService worker starting...")
        try:
            self._ensure_stream_is_open() # Initial attempt to open the stream
        except Exception as e:
            self.logger.error(f"AudioStreamService could not start stream: {e}. Worker will not run.")
            return

        audio_queue = self.shared_resources['queues']['audio_output_queue']
        playback_buffer = []
        # Configurable: number of chunks to buffer before playing, or total bytes, or time duration.
        # For simplicity, let's buffer a fixed number of chunks.
        # This value could come from config.py or shared_resources.
        buffer_chunk_count = self.shared_resources.get('config', {}).get('audio_buffer_chunk_count', 2) 
        self.logger.info(f"AudioStreamService will buffer {buffer_chunk_count} chunks before playback.")

        while True:
            try:
                # Try to get a chunk from the queue
                try:
                    wav_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.1) # Short timeout to allow buffer processing
                except asyncio.TimeoutError:
                    # If queue is empty and buffer has content, play it
                    if playback_buffer:
                        self.logger.debug(f"Audio queue empty, playing {len(playback_buffer)} buffered chunks.")
                        await self._play_buffered_audio(playback_buffer)
                    await asyncio.sleep(0.05) # Brief sleep if queue was empty and buffer was also empty
                    continue # Go back to check queue

                if wav_bytes is None:  # Sentinel value to stop the worker
                    self.logger.info("Received None. Playing any remaining buffered audio and stopping worker.")
                    self._play_buffered_audio(playback_buffer)
                    break # Exit the main while loop

                playback_buffer.append(wav_bytes)
                audio_queue.task_done()

                if len(playback_buffer) >= buffer_chunk_count:
                    self.logger.debug(f"Buffer full ({len(playback_buffer)} chunks), playing audio.")
                    await self._play_buffered_audio(playback_buffer)

            except asyncio.CancelledError:
                self.logger.info("AudioStreamService worker cancelled. Playing any remaining buffered audio.")
                if playback_buffer: # Ensure buffer is not empty before attempting to play
                    await self._play_buffered_audio(playback_buffer)
                break
            except Exception as e:
                self.logger.error(f"Error in AudioStreamService worker loop: {e}")
                # Potentially attempt to reopen stream or pause before retrying
                await asyncio.sleep(1) # Wait a bit before retrying to avoid busy loop on persistent errors
                try:
                    self._ensure_stream_is_open() # Try to recover the stream
                except Exception as rec_e:
                    self.logger.error(f"Failed to recover audio stream: {rec_e}. Worker might be stuck.")
                    # If stream recovery fails repeatedly, might need to stop the worker.
                    # Consider a retry limit or a more robust recovery mechanism.
        
        # if prefetch_task_handle:
        #     prefetch_task_handle.cancel()
        #     try:
        #         await prefetch_task_handle
        #     except asyncio.CancelledError:
        #         self.logger.info("Prefetch task successfully cancelled on worker exit.")

        self.logger.info("AudioStreamService worker finished.")

    async def stop_service(self):
        """Override to include specific cleanup for this service."""
        self.logger.info("Stopping AudioStreamService...")
        # Signal the run_worker loop to terminate by putting None in the queue
        audio_queue = self.shared_resources['queues']['audio_output_queue']
        await audio_queue.put(None)
        
        # Call superclass stop_service for general cancellation
        await super().stop_service()
        
        # Cleanup playback backend
        if hasattr(self.audio_playback_backend, 'cleanup') and callable(self.audio_playback_backend.cleanup):
            self.audio_playback_backend.cleanup()
        elif hasattr(self.audio_playback_backend, 'close_stream') and callable(self.audio_playback_backend.close_stream):
            self.audio_playback_backend.close_stream() # Fallback if no cleanup method
        self.logger.info("AudioStreamService stopped and backend cleaned up.")

    def __del__(self):
        # This is a fallback. Explicit cleanup via stop_service is preferred.
        self.logger.info("AudioStreamService __del__ called.")
        if hasattr(self.audio_playback_backend, 'cleanup') and callable(self.audio_playback_backend.cleanup):
            self.audio_playback_backend.cleanup()
        elif hasattr(self.audio_playback_backend, 'close_stream') and callable(self.audio_playback_backend.close_stream):
             # Ensure that close_stream is not an async method if called from __del__
            try:
                # Check if it's an async method; __del__ cannot await
                if asyncio.iscoroutinefunction(self.audio_playback_backend.close_stream):
                    self.logger.warning("Cannot call async close_stream from __del__. Backend may not be cleaned properly.")
                else:
                    self.audio_playback_backend.close_stream()
            except Exception as e:
                self.logger.error(f"Error during __del__ backend cleanup: {e}")