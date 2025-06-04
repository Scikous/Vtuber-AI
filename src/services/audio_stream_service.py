import asyncio
from typing import Optional
from .base_service import BaseService
from TTS_Wizard.utils.audio_playback_base import AudioPlaybackBase
from TTS_Wizard.utils.pyaudio_playback import PyAudioPlayback # Default backend

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
        self._playback_paused_event = asyncio.Event() # Internal event to signal worker to pause processing
        self._service_stop_event = asyncio.Event() # Event to signal the worker to stop completely
        self._chunk_size = 512
        # Shared state events (to be managed/set externally)
        self.terminate_current_dialogue_event = shared_resources.get("terminate_current_dialogue_event", asyncio.Event()) # Stops current dialogue playback
        self.is_audio_streaming_event = shared_resources.get("is_audio_streaming_event", asyncio.Event()) # Pauses playback when user speaks

    def _ensure_stream_is_open(self):
        """Ensures the audio stream is open, attempting to open it if not."""
        try:
            # is_active implies open and not paused by the backend's own controls.
            # If our service wants to play, and the backend isn't paused by us, and it's not active, open it.
            if not self.audio_playback_backend.is_active() and not self.audio_playback_backend.is_paused():
                self.logger.info("Audio stream is not active or not paused by backend. Attempting to open...")
                self.audio_playback_backend.open_stream()
            elif self.audio_playback_backend.is_paused():
                self.logger.info("Audio stream is currently paused by backend. Will resume when appropriate.")
        except Exception as e:
            self.logger.error(f"Failed to open or ensure audio stream is active: {e}")
            raise 

    async def _play_chunk(self, chunk: bytes):
        if self._playback_paused_event.is_set():
            self.logger.debug("Playback is paused, not playing chunk now.")
            return

        try:
            self._ensure_stream_is_open() 
            if self.audio_playback_backend.is_paused():
                self.logger.info("Playback backend is paused, attempting to resume for playing chunk.")
                self.audio_playback_backend.resume_stream()
                
            if not self.audio_playback_backend.is_active():
                self.logger.warning("Audio stream not active after attempting to open/resume. Cannot play.")
                return

            self.is_audio_streaming_event.set()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.audio_playback_backend.write_chunk, chunk)
            self.logger.debug(f"Played audio chunk.")
            self.is_audio_streaming_event.clear()
        except Exception as e:
            self.logger.error(f"Error playing audio chunk: {e}")

    async def run_worker(self):
        self.logger.info("AudioStreamService worker starting...")
        try:
            self._ensure_stream_is_open() 
        except Exception as e:
            self.logger.error(f"AudioStreamService could not start stream: {e}. Worker will not run.")
            return

        audio_queue = self.shared_resources['queues']['audio_output_queue']
        self.logger.info("AudioStreamService will feed directly from queue to playback.")

        self._service_stop_event.clear() # Ensure stop event is clear at start
        self._playback_paused_event.clear() # Ensure pause event is clear at start

        while not self._service_stop_event.is_set():
            try:
                # Handle pause state from internal command or external event (user speaking)
                if self._playback_paused_event.is_set():
                    if not self.audio_playback_backend.is_paused():
                        self.logger.info("AudioStreamService: Playback pause detected. Pausing backend stream.")
                        self.audio_playback_backend.pause_stream()
                    await asyncio.sleep(0.1) # Sleep while paused
                    continue
                else: # Not paused by service
                    if self.audio_playback_backend.is_paused():
                        self.logger.info("AudioStreamService: Playback resume detected. Resuming backend stream.")
                        self.audio_playback_backend.resume_stream()
                wav_bytes = None
                try:
                    wav_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.01) # Shorter timeout for more frequent checks
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.01) # Sleep briefly before checking again if the queue is empty
                    continue

                if wav_bytes is None:  # Sentinel value to stop the worker
                    self.logger.info("Received None sentinel. Stopping worker.")
                    self._service_stop_event.set() # Signal worker to stop
                    break
                while wav_bytes:
                    if await self.check_terminate_event(audio_queue=audio_queue): break
                    audio_chunk = wav_bytes[:self._chunk_size]
                    await self._play_chunk(audio_chunk)
                    wav_bytes = wav_bytes[self._chunk_size:]
                audio_queue.task_done()

            except asyncio.CancelledError:
                self.logger.info("AudioStreamService worker run_worker task cancelled.")
                self._service_stop_event.set() # Ensure loop terminates
                break
            except Exception as e:
                self.logger.error(f"Error in AudioStreamService worker loop: {e}", exc_info=True)
                await asyncio.sleep(1)
                try:
                    self._ensure_stream_is_open()
                except Exception as rec_e:
                    self.logger.error(f"Failed to recover audio stream: {rec_e}. Worker might be stuck.")
        self.logger.info("AudioStreamService worker finished processing loop.")
        # Final cleanup of the backend stream when worker stops for good
        if self.audio_playback_backend:
            self.audio_playback_backend.close_stream()
            self.logger.info("Audio playback backend stream closed by worker exit.")


    async def check_terminate_event(self, audio_queue):
        # Handle termination of current dialogue
        if self.terminate_current_dialogue_event.is_set():
            self.logger.info("AudioStreamService: Terminate current dialogue event detected.")
            # Drain the audio_output_queue of any remaining chunks for this dialogue.
            self.audio_playback_backend.stop_and_clear_internal_buffers() # Stop any sound already in hardware buffer
            while not audio_queue.empty():
                try:
                    item = audio_queue.get_nowait()
                    audio_queue.task_done()
                    self.logger.debug(f"Discarded audio chunk from queue due to termination.")
                except asyncio.QueueEmpty:
                    break
            self.terminate_current_dialogue_event.clear() # Reset the event
            self.is_audio_streaming_event.clear() # Signal that audio is streaming
            self.logger.info("Audio playback for current dialogue terminated and buffer/queue cleared.")
            return True
        return False
            # await asyncio.sleep(0.1) # Sleep briefly before checking for new audio/pause states.
            

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