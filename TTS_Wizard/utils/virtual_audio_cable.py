"""
Virtual Audio Cable Implementation for Python
Supports audio routing, capture, and forwarding between different sources and destinations.
"""

import pyaudio
import threading
import queue
import numpy as np
import time
import asyncio
from typing import Optional, Callable, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualAudioCable:
    """
    Virtual Audio Cable implementation that can route audio between different sources and sinks.
    Supports real-time audio processing and forwarding.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 channels: int = 2,
                 format_type: int = pyaudio.paFloat32):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        
        # Audio interface
        self.audio = pyaudio.PyAudio()
        
        # Audio buffers and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.output_queues: Dict[str, queue.Queue] = {}
        
        # Threading
        self.running = False
        self.threads = []
        
        # Audio processing callbacks
        self.audio_processors: List[Callable] = []
        
        # Device info
        self.input_devices = self._get_input_devices()
        self.output_devices = self._get_output_devices()
        
    def _get_input_devices(self) -> List[Dict]:
        """Get available input audio devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': info['defaultSampleRate']
                })
        return devices
    
    def _get_output_devices(self) -> List[Dict]:
        """Get available output audio devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxOutputChannels'],
                    'sample_rate': info['defaultSampleRate']
                })
        return devices
    
    def list_devices(self):
        """Print available audio devices"""
        print("=== INPUT DEVICES ===")
        for device in self.input_devices:
            print(f"Index: {device['index']}, Name: {device['name']}, Channels: {device['channels']}")
        
        print("\n=== OUTPUT DEVICES ===")
        for device in self.output_devices:
            print(f"Index: {device['index']}, Name: {device['name']}, Channels: {device['channels']}")
    
    def add_audio_processor(self, processor: Callable):
        """Add an audio processing function that will be applied to all audio data"""
        self.audio_processors.append(processor)
    
    def create_output_channel(self, channel_name: str) -> queue.Queue:
        """Create a new output channel that can receive audio data"""
        self.output_queues[channel_name] = queue.Queue(maxsize=50)
        return self.output_queues[channel_name]
    
    def start_audio_capture(self, input_device_index: Optional[int] = None):
        """Start capturing audio from specified input device"""
        def audio_capture_thread():
            try:
                stream = self.audio.open(
                    format=self.format_type,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=input_device_index,
                    frames_per_buffer=self.chunk_size
                )
                
                logger.info(f"Started audio capture from device {input_device_index}")
                
                while self.running:
                    try:
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        audio_array = np.frombuffer(data, dtype=np.float32)
                        
                        # Apply audio processors
                        for processor in self.audio_processors:
                            audio_array = processor(audio_array)
                        
                        # Put processed audio in main queue
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio_array.tobytes())
                        
                        # Distribute to output channels
                        for channel_queue in self.output_queues.values():
                            if not channel_queue.full():
                                channel_queue.put(audio_array.copy())
                        
                    except Exception as e:
                        logger.error(f"Error in audio capture: {e}")
                        time.sleep(0.01)
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Failed to start audio capture: {e}")
        
        thread = threading.Thread(target=audio_capture_thread, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def start_audio_output(self, output_device_index: Optional[int] = None, channel_name: str = "main"):
        """Start playing audio to specified output device"""
        if channel_name not in self.output_queues:
            self.create_output_channel(channel_name)
        
        def audio_output_thread():
            try:
                stream = self.audio.open(
                    format=self.format_type,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=output_device_index,
                    frames_per_buffer=self.chunk_size
                )
                
                logger.info(f"Started audio output to device {output_device_index}")
                
                while self.running:
                    try:
                        audio_data = self.output_queues[channel_name].get(timeout=0.1)
                        stream.write(audio_data.tobytes())
                    except queue.Empty:
                        # Write silence to prevent buffer underrun
                        silence = np.zeros(self.chunk_size * self.channels, dtype=np.float32)
                        stream.write(silence.tobytes())
                    except Exception as e:
                        logger.error(f"Error in audio output: {e}")
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Failed to start audio output: {e}")
        
        thread = threading.Thread(target=audio_output_thread, daemon=True)
        self.threads.append(thread)
        thread.start()
    
    def start_virtual_cable(self, 
                           input_device_index: Optional[int] = None,
                           output_device_index: Optional[int] = None):
        """Start the virtual audio cable (input -> processing -> output)"""
        self.running = True
        
        # Start input capture
        self.start_audio_capture(input_device_index)
        
        # Start output playback
        self.start_audio_output(output_device_index)
        
        logger.info("Virtual audio cable started")
    
    def send_audio_to_channel(self, audio_data: np.ndarray, channel_name: str):
        """Send audio data to a specific output channel"""
        if channel_name in self.output_queues:
            if not self.output_queues[channel_name].full():
                self.output_queues[channel_name].put(audio_data)
    
    def get_audio_data(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio data from the main queue"""
        try:
            data = self.audio_queue.get(timeout=timeout)
            return np.frombuffer(data, dtype=np.float32)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the virtual audio cable"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        for queue_obj in self.output_queues.values():
            while not queue_obj.empty():
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Virtual audio cable stopped")
    
    def __del__(self):
        if hasattr(self, 'audio'):
            self.audio.terminate()


class AudioEffects:
    """Collection of audio processing effects"""
    
    @staticmethod
    def volume_control(audio_data: np.ndarray, volume: float = 1.0) -> np.ndarray:
        """Adjust audio volume"""
        return audio_data * volume
    
    @staticmethod
    def noise_gate(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Simple noise gate"""
        mask = np.abs(audio_data) > threshold
        return audio_data * mask
    
    @staticmethod
    def simple_reverb(audio_data: np.ndarray, delay_samples: int = 1000, decay: float = 0.3) -> np.ndarray:
        """Simple reverb effect"""
        if len(audio_data) <= delay_samples:
            return audio_data
        
        reverb_data = audio_data.copy()
        reverb_data[delay_samples:] += audio_data[:-delay_samples] * decay
        return reverb_data


# Example usage and STT integration
class STTIntegration:
    """Speech-to-Text integration for captured audio"""
    
    def __init__(self, vac: VirtualAudioCable):
        self.vac = vac
        self.stt_queue = queue.Queue()
        self.running = False
        
    def start_stt_capture(self):
        """Start capturing audio for STT processing"""
        self.running = True
        
        def stt_worker():
            while self.running:
                audio_data = self.vac.get_audio_data(timeout=0.5)
                if audio_data is not None:
                    # Here you would integrate with your STT service
                    # For example, with speech_recognition, whisper, etc.
                    self.process_stt_audio(audio_data)
        
        thread = threading.Thread(target=stt_worker, daemon=True)
        thread.start()
    
    def process_stt_audio(self, audio_data: np.ndarray):
        """Process audio data for STT (placeholder for actual STT integration)"""
        # This is where you'd integrate with your STT system
        # Example integrations:
        # - OpenAI Whisper
        # - Google Speech-to-Text
        # - Azure Speech Services
        # - Local STT models
        pass
    
    def stop(self):
        self.running = False


# Example usage
if __name__ == "__main__":
    # Create virtual audio cable
    vac = VirtualAudioCable(sample_rate=44100, chunk_size=1024, channels=2)
    
    # List available devices
    vac.list_devices()
    
    # Add audio effects
    vac.add_audio_processor(lambda audio: AudioEffects.volume_control(audio, 1.2))
    vac.add_audio_processor(lambda audio: AudioEffects.noise_gate(audio, 0.02))
    
    # Create additional output channels
    discord_channel = vac.create_output_channel("discord")
    obs_channel = vac.create_output_channel("obs")
    
    # Start the virtual cable
    # Replace None with specific device indices if needed
    vac.start_virtual_cable(input_device_index=None, output_device_index=None)
    
    # Initialize STT integration
    stt = STTIntegration(vac)
    stt.start_stt_capture()
    
    try:
        print("Virtual Audio Cable running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        vac.stop()
        stt.stop()
