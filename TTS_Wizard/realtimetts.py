import time
import logging
from TTS_Wizard.RealtimeTTS.RealtimeTTS.engines import CoquiEngine
from TTS_Wizard.RealtimeTTS.RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice
from TTS_Wizard.utils.tts_base import TTSBase
from typing import Any, Optional, Callable

class RealTimeTTS(TTSBase):
    """
    A real-time Text-to-Speech class that is engine-agnostic.

    This class takes a pre-configured TTS engine and handles the real-time
    streaming, playback, and event management.
    """
    def __init__(self,
                tts_engine: Any,
                stream_options: Optional[dict]=None,
                tts_playback_approved_event: Optional[Any]=None,
                **kwargs: Any):
        """
        Initializes the RealtimeTTS system.

        Args:
            tts_engine: A pre-initialized and configured TTS engine instance
                        (e.g., CoquiEngine, PiperEngine).
            stream_options (dict, optional): Options for the TextToAudioStream.
            **kwargs: Supports additional arguments, such as events for synchronization.
                    - tts_playback_approved_event: An asyncio.Event to gate playback.
        """
        super().__init__(**kwargs)
        self.engine = tts_engine
        stream_options = stream_options or {}

        # Event from an orchestrator
        self.tts_playback_approved_event = tts_playback_approved_event #kwargs.get("tts_playback_approved_event")

        # Internal state for performance metrics
        self.start_time: Optional[float] = None

        # Optimized stream arguments for ultra-low latency
        stream_args = {
            'on_audio_stream_start': self._on_audio_stream_start_callback,
            'on_audio_stream_stop': self._on_audio_stream_stop_callback,
            'playout_chunk_size': 64,
            'frames_per_buffer': 64,
            'log_characters': False,
            **stream_options  # Allow user to override defaults
        }

        self.stream = TextToAudioStream(self.engine, **stream_args)
        self._warm_up_engine()
        
        # Create callbacks for synthesis lifecycle
        self._before_sentence_callback, self._on_sentence_callback, self._on_audio_chunk = self._create_synthesis_callbacks()

    def _warm_up_engine(self):
            """Warms up the TTS engine to prevent delays on the first synthesis."""
            self.logger.info("Warming up TTS engine...")
            self.stream.feed("warm up").play_async(muted=True)
            self.logger.info("TTS engine ready.")

    def _create_synthesis_callbacks(self) -> tuple[Callable, Callable, Callable]:
        """Creates and returns the callbacks for handling synthesis events."""
        sentence_synth_start: Optional[float] = None

        def before_sentence_callback(_):
            nonlocal sentence_synth_start
            if self.start_time:
                sentence_synth_start = time.time()
                elapsed = sentence_synth_start - self.start_time
                self.logger.debug(f"<SYNTHESIS_START> {elapsed:.2f}s")

        def on_sentence_callback(_):
            nonlocal sentence_synth_start
            if sentence_synth_start:
                delta = time.time() - sentence_synth_start
                self.logger.debug(f"<SYNTHESIS_DONE> Delta: {delta:.2f}s")
            else:
                self.logger.debug("<SYNTHESIS_DONE> No start time recorded.")

        def on_audio_chunk(_):
            # Gate playback based on the tts_playback_approved_event
            if self.tts_playback_approved_event and not self.tts_playback_approved_event.is_set():
                self.logger.info("User is speaking, holding TTS playback...")
                self.tts_playback_approved_event.wait()  # Block until event is set
        return before_sentence_callback, on_sentence_callback, on_audio_chunk

    def _on_audio_stream_start_callback(self):
        """Callback for when the first audio chunk is ready to play."""
        if self.start_time:
            delta = time.time() - self.start_time
            self.logger.debug(f"<TTFT> Time to first audio: {delta:.2f}s")

    def _on_audio_stream_stop_callback(self):
        """Callback for when the audio stream has fully stopped."""
        self.logger.info("Audio stream stopped.")

    async def stop(self):
        """Stops the audio stream if it is currently playing."""
        if self.stream.is_playing():
            self.logger.info("Stopping audio stream...")
            self.stream.stop()

    async def speak(self, text: str, **kwargs: Any):
        """
        Feeds text to the stream and starts playback asynchronously.

        Args:
            text (str): The text to be synthesized.
            **kwargs: Additional options for playback.
                      - min_sentence_len: Minimum length for a sentence fragment.
        """
        if not self.stream.is_playing():
            self.start_time = time.time()

        self.stream.feed(text)

        if not self.stream.is_playing():
            self.logger.info("🚀 Starting ultra-fast audio stream...")
            self.stream.play_async(
                log_synthesized_text=False,
                before_sentence_synthesized=self._before_sentence_callback,
                on_sentence_synthesized=self._on_sentence_callback,
                on_audio_chunk=self._on_audio_chunk,
                fast_sentence_fragment=True,
                minimum_sentence_length=kwargs.get('min_sentence_len', 8),
                force_first_fragment_after_words=6,
            )

    def shutdown(self):
        """Shuts down the underlying TTS engine and releases resources."""
        self.logger.info("Shutting down TTS engine.")
        self.engine.shutdown()


def pipertts_engine(model_file, config_file, piper_path):
    """
    Initializes and configures the Piper TTS engine.

    Args:
        model_file (str): Path to the model file.
        config_file (str): Path to the configuration file.
        piper_path (str): Path to the Piper executable.
        voice (str): The voice to be used.
        **tts_settings: Additional settings for the TTS engine.

    Returns:
        PiperEngine: An initialized and configured PiperEngine instance.
    """
    # Initialize and configure the Piper TTS engine
    piper_voice = PiperVoice(
        model_file=model_file,
        config_file=config_file
    )
    piper_tts_engine = PiperEngine(
        piper_path=piper_path,
        voice=piper_voice,
    )
    return piper_tts_engine

def coquitts_engine(use_deepspeed: bool=True):
    """Create optimized Coqui engine for ultra-low latency."""
    return CoquiEngine(
        use_deepspeed=use_deepspeed,
        stream_chunk_size=8,        # Reduced for faster streaming
        overlap_wav_len=512,        # Reduced overlap for speed
        temperature=0.7,            # Balanced quality/speed
        length_penalty=0.8,         # Faster generation
        repetition_penalty=5.0,     # Prevent repetition
        top_k=50,                   # Reduced for speed
        top_p=0.85,                 # Balanced quality/speed
        enable_text_splitting=True, # Enable for streaming
        thread_count=6,             # Optimize for multi-core
        device="cuda",              # Force CUDA
        level=40                    # Reduce logging
    )

# if __name__ == "__main__":


#     tts_engine = coquitts_engine()
#     realtime_tts = RealTimeTTS(tts_engine, {})