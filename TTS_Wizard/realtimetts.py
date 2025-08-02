import time
import asyncio
from TTS_Wizard.RealtimeTTS.RealtimeTTS.engines import CoquiEngine
from TTS_Wizard.RealtimeTTS.RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice
import threading

class RealTimeTTS():
    """
    A real-time Text-to-Speech class that is engine-agnostic.

    This class takes a pre-configured TTS engine and handles the real-time
    streaming, playback, and event management.
    """
    def __init__(self, tts_engine, stream_options={}, **kwargs):
        """
        Initializes the RealTimeTTS system.

        Args:
            tts_engine: A pre-initialized and configured TTS engine instance 
                        (e.g., CoquiEngine, PiperEngine).
            stream_options (dict, optional): A dictionary of options to be passed 
                                             to the TextToAudioStream. Defaults to None.
            **kwargs: Additional keyword arguments. -- usually an event.
        """
        self.engine = tts_engine
        
        # This event is passed from the orchestrator, controlled by the STT VAD
        self.user_has_stopped_speaking_event = kwargs.get("user_has_stopped_speaking_event")
        
        # Internal state for timing
        self.start_time = None
        
        # Prepare arguments for the stream with ultra-low latency settings
        stream_args = {
            'on_audio_stream_start': self.on_audio_stream_start_callback,
            'on_audio_stream_stop': self.on_audio_stream_stop_callback,
            'playout_chunk_size': stream_options.get('playout_chunk_size', 64),  # Reduced for lower latency
            'frames_per_buffer': stream_options.get('frames_per_buffer', 64),    # Reduced for lower latency
            'muted': False,
            'log_characters': False  # Disable for speed
        }
        if stream_options:
            stream_args.update(stream_options)

        # Initialize the stream with the configured engine and arguments
        self.stream = TextToAudioStream(self.engine, **stream_args)
        
        print("Warming up TTS engine...")
        self.stream.feed("warm up").play_async(muted=True)
        print("TTS engine ready.")
        
        self.before_sentence_callback, self.on_sentence_callback, self.on_audio_chunk = self.create_synthesis_callbacks()

    def create_synthesis_callbacks(self):
        sentence_synth_start = None

        def before_sentence_callback(_):
            nonlocal sentence_synth_start
            if self.start_time:
                sentence_synth_start = time.time()
                elapsed = sentence_synth_start - self.start_time
                print(f"<SYNTHESIS_START> {elapsed:.2f}s")

        def on_sentence_callback(_):
            if sentence_synth_start is not None:
                delta = time.time() - sentence_synth_start
                print(f"<SYNTHESIS_DONE> Delta: {delta:.2f}s")
            else:
                print("<SYNTHESIS_DONE> No start time recorded.")

        def on_audio_chunk(_):
            # Gate playback based on user speaking state
            if self.user_has_stopped_speaking_event and not self.user_has_stopped_speaking_event.is_set():
                print("User is speaking, holding TTS playback...")
                self.user_has_stopped_speaking_event.wait() # Block until STT sets the event

        return before_sentence_callback, on_sentence_callback, on_audio_chunk

    def cleanup(self):
        """Shuts down the underlying TTS engine."""
        print("Shutting down TTS engine.")
        self.engine.shutdown()

    def on_audio_stream_start_callback(self):
        if self.start_time:
            delta = time.time() - self.start_time
            print(f"<TTFT> Time to first audio: {delta:.2f}s")
        
    def on_audio_stream_stop_callback(self):
        print("Audio stream stopped.")

    async def tts_request_clear(self):
        if self.stream.is_playing():
            self.stream.stop()


    async def tts_request_async(self, text, min_sentence_len=8, **kwargs):
        """
        Feeds text to the stream and starts playback asynchronously.
        If playback is already in progress, it simply queues the text.
        """
        # Set the start time for this specific TTS request for accurate metrics
        if not self.stream.is_playing():
             self.start_time = time.time()

        self.stream.feed(text)

        if not self.stream.is_playing():
            # print("🚀 Ready for ultra-fast audio stream...")
            print("🚀 Starting ultra-fast audio stream...")
            
            self.stream.play_async(
                log_synthesized_text=False,  # Disabled for speed
                before_sentence_synthesized=self.before_sentence_callback,
                on_sentence_synthesized=self.on_sentence_callback,
                on_audio_chunk=self.on_audio_chunk,
                fast_sentence_fragment=True,
                minimum_sentence_length=min_sentence_len,
                force_first_fragment_after_words=6,  # Reduced for faster response
                muted=False
            )



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

def coquitts_engine(use_deepspeed=True):
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