import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def list_audio_devices():
    """Lists all available audio input devices."""
    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get('maxInputChannels') > 0:
            print(f"  Device ID {i}: {dev_info.get('name')}")
    p.terminate()

def get_audio_data(device_index=None, duration=5, output_filename="output.wav"):
    """
    Records audio from a specified input device (or default) for a given duration 
    and saves it to a WAV file.

    Args:
        device_index (int, optional): The index of the audio input device.
                                      If None, the default input device is used.
        duration (int, optional): The duration of the recording in seconds. Defaults to 5.
        output_filename (str, optional): The name of the output WAV file. 
                                         Defaults to "output.wav".

    Returns:
        bytes: The recorded audio data, or None if recording fails.
    """
    p = pyaudio.PyAudio()

    if device_index is None:
        try:
            device_index = p.get_default_input_device_info()['index']
            print(f"Using default input device: {p.get_device_info_by_index(device_index)['name']}")
        except IOError:
            print("Error: No default input device found. Please specify a device index.")
            p.terminate()
            return None
    else:
        try:
            device_info = p.get_device_info_by_index(device_index)
            if device_info.get('maxInputChannels') == 0:
                print(f"Error: Device ID {device_index} is not an input device.")
                p.terminate()
                return None
            print(f"Using input device: {device_info['name']}")
        except IndexError:
            print(f"Error: Device ID {device_index} not found.")
            p.terminate()
            return None

    stream = None
    frames = []

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=device_index)

        print("Recording...")

        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Finished recording.")

    except Exception as e:
        print(f"Error during recording: {e}")
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        return None
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {output_filename}")

    return b''.join(frames)

if __name__ == '__main__':
    list_audio_devices()
    # Example usage: Record 5 seconds from the default device
    audio_data = get_audio_data(duration=5)
    if audio_data:
        print(f"Recorded {len(audio_data)} bytes of audio.")

    # Example usage: Record 5 seconds from a specific device (e.g., device ID 1)
    # Make sure to replace 1 with an actual input device ID from the list_audio_devices() output
    # audio_data_specific = get_audio_data(device_index=1, duration=5, output_filename="output_specific_device.wav")
    # if audio_data_specific:
    #     print(f"Recorded {len(audio_data_specific)} bytes of audio from specific device.")