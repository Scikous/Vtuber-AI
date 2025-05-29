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





    async def _play_buffered_audio(self, playback_buffer: list):
        """Plays the content of the playback_buffer and clears it asynchronously."""
        if not playback_buffer:
            return

        if self._playback_paused_event.is_set():
            self.logger.debug("Playback is paused, not playing buffered audio now.")
            return # Don't play if paused

        try:
            self._ensure_stream_is_open() # Ensure stream is ready
            if self.audio_playback_backend.is_paused(): # If backend was paused by us/external
                self.logger.info("Playback backend is paused, attempting to resume for playing buffer.")
                self.audio_playback_backend.resume_stream()

            if not self.audio_playback_backend.is_active():
                self.logger.warning("Audio stream not active after attempting to open/resume. Cannot play.")
                return

            self.is_audio_streaming_event.set() # Signal that audio is streaming
            full_audio_data = b''.join(chunk for chunk in playback_buffer)
            if full_audio_data:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.audio_playback_backend.write_chunk, full_audio_data)
                self.logger.debug(f"Played {len(playback_buffer)} buffered audio chunks.")
            playback_buffer.clear()
            self.is_audio_streaming_event.clear() # Signal that audio is streaming

        except Exception as e:
            self.logger.error(f"Error playing buffered audio: {e}")
            # Consider how to handle errors: clear buffer? try again?
            playback_buffer.clear() # Clear buffer on error to avoid replaying bad data


    async def run_worker(self):
        self.logger.info("AudioStreamService worker starting...")
        try:
            self._ensure_stream_is_open() 
        except Exception as e:
            self.logger.error(f"AudioStreamService could not start stream: {e}. Worker will not run.")
            return

        audio_queue = self.shared_resources['queues']['audio_output_queue']
        playback_buffer = []
        buffer_chunk_count = self.shared_resources.get('config', {}).get('audio_buffer_chunk_count', 2) 
        self.logger.info(f"AudioStreamService will buffer {buffer_chunk_count} chunks before playback.")

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
                
                # Handle termination of current dialogue
                if self.terminate_current_dialogue_event.is_set():
                    self.logger.info("AudioStreamService: Terminate current dialogue event detected.")
                    if playback_buffer:
                        self.logger.info(f"Clearing {len(playback_buffer)} chunks from playback_buffer due to dialogue termination.")
                        playback_buffer.clear()
                    # Drain the audio_output_queue of any remaining chunks for this dialogue.
                    # This assumes that once terminate is set, new audio from TTS for this dialogue will stop.
                    # A more robust system might use utterance IDs to clear specific items.
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
                    await asyncio.sleep(0.1) # Sleep briefly before checking for new audio/pause states.
                    continue # Skip rest of loop, don't try to get audio from queue.
                    # After termination, continue to next iteration to check for new audio/pause states.

                # Try to get a chunk from the queue
                wav_bytes = None
                try:
                    wav_bytes = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Queue empty: if buffer has content and we are not paused, play it
                    if playback_buffer and not (self._playback_paused_event.is_set()):
                        self.logger.debug(f"Audio queue empty, playing {len(playback_buffer)} buffered chunks.")
                        await self._play_buffered_audio(playback_buffer)
                    await asyncio.sleep(0.05) 
                    continue 

                if wav_bytes is None:  # Sentinel value to stop the worker
                    self.logger.info("Received None sentinel. Playing remaining buffered audio and stopping worker.")
                    self._service_stop_event.set() # Signal worker to stop
                    if playback_buffer: # Play any remaining before full stop
                        await self._play_buffered_audio(playback_buffer) # This will respect pause state
                    break 

                playback_buffer.append(wav_bytes)
                audio_queue.task_done()

                if len(playback_buffer) >= buffer_chunk_count:
                    self.logger.debug(f"Buffer full ({len(playback_buffer)} chunks), attempting to play audio.")
                    if not (self._playback_paused_event.is_set()):
                        await self._play_buffered_audio(playback_buffer)
                    else:
                        self.logger.debug("Buffer full, but playback is paused. Holding chunks.")
                
            except asyncio.CancelledError:
                self.logger.info("AudioStreamService worker run_worker task cancelled.")
                self._service_stop_event.set() # Ensure loop terminates
                if playback_buffer: 
                    # Potentially play out buffer if not paused, or just log. For now, log.
                    self.logger.info(f"Run_worker cancelled, {len(playback_buffer)} chunks in buffer.")
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

    # --- Service Control Methods ---
    async def pause_playback(self):
        """Externally called method to pause audio playback."""
        if not self._playback_paused_event.is_set():
            self.logger.info("AudioStreamService: Pausing playback via external command.")
            self._playback_paused_event.set()
            # The worker loop will detect this and call backend's pause_stream
        else:
            self.logger.info("AudioStreamService: Playback already paused by internal command.")

    async def resume_playback(self):
        """Externally called method to resume audio playback."""
        if self._playback_paused_event.is_set():
            self.logger.info("AudioStreamService: Resuming playback via external command.")
            self._playback_paused_event.clear()
            # The worker loop will detect this and call backend's resume_stream
            # Also clear user_speaking_pause_event if it was the cause, or let external logic handle it.
            # For now, this resume overrides user_speaking_pause_event.
            # A more nuanced approach might be needed if multiple pause sources exist.
            # if self.user_speaking_pause_event.is_set(): # Removed this block
            #     self.logger.info("AudioStreamService: Resuming also clears user_speaking_pause_event.")
                # self.user_speaking_pause_event.clear() # Decided against this, let external logic manage user_speaking_pause_event
        else:
            self.logger.info("AudioStreamService: Playback not paused by internal command, no resume action needed.")

    async def stop_current_audio_output(self, clear_buffer: bool = True, clear_queue: bool = True):
        """
        Externally called method to stop current audio output immediately.
        This is more forceful than pause. It clears buffers and stops the hardware.
        """
        self.logger.info("AudioStreamService: Stopping current audio output immediately.")
        if self.audio_playback_backend:
            self.audio_playback_backend.stop_and_clear_internal_buffers()
        
        if clear_buffer and hasattr(self, 'playback_buffer'): # playback_buffer is local to run_worker
            # This method can't directly clear playback_buffer as it's in run_worker's scope.
            # Instead, we signal the worker. The terminate_current_dialogue_event serves this purpose.
            self.logger.info("Setting terminate_current_dialogue_event to clear buffers in worker.")
            self.terminate_current_dialogue_event.set() 
            # The worker loop will see this and clear its buffer and the queue.
        
        # If we need to clear the queue directly here (e.g. if worker is slow to respond)
        if clear_queue:
            audio_queue = self.shared_resources['queues']['audio_output_queue']
            drained_count = 0
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                    drained_count +=1
                except asyncio.QueueEmpty:
                    break
            if drained_count > 0:
                self.logger.info(f"Drained {drained_count} items from audio_output_queue directly.")


    async def stop_service(self):
        """Override to include specific cleanup for this service."""
        self.logger.info("Stopping AudioStreamService...")
        self._service_stop_event.set() # Signal the run_worker loop to terminate

        # Put None in the queue to ensure worker loop can exit if waiting on queue.get()
        audio_queue = self.shared_resources['queues']['audio_output_queue']
        await audio_queue.put(None) 
        
        await super().stop_service() # This will cancel the worker task
        
        # Wait for worker task to finish (already handled by super if worker_task is managed there)
        # if self.worker_task:
        #     try:
        #         await self.worker_task
        #     except asyncio.CancelledError:
        #         self.logger.info("AudioStreamService worker task was cancelled as part of stop_service.")
        
        # Final cleanup of playback backend
        if self.audio_playback_backend:
            if hasattr(self.audio_playback_backend, 'cleanup') and callable(self.audio_playback_backend.cleanup):
                self.audio_playback_backend.cleanup()
                self.logger.info("Audio playback backend cleaned up.")
            elif hasattr(self.audio_playback_backend, 'close_stream') and callable(self.audio_playback_backend.close_stream):
                self.audio_playback_backend.close_stream() 
                self.logger.info("Audio playback backend stream closed via close_stream().")
        self.logger.info("AudioStreamService stopped.")


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