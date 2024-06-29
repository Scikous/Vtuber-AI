
from gradio_client import Client
import json, requests
import pyaudio
from pynput import keyboard
from queue import Queue
import threading
import time


tts_queue = Queue(maxsize=2)

#prompt_text example, causes issues: But truly, is a simple piece of paper worth the credit people give it?
def send_tts_request(text="(Super Elite Magnificent Agent John Smith!)", text_lang="en",
                        ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav",
                          prompt_text="", prompt_lang="en",
                          top_k=7, top_p=.87, temperature=0.87,
                          text_split_method="cut5",
                          batch_size=1, batch_threshold=0.75, split_bucket=True,
                          speed_factor=1.0, fragment_interval=0.3,
                          seed= -1,
                          media_type="wav",
                          streaming_mode=False, parallel_infer=True,
                          repetition_penalty=1.35
                          ):
    """
    Sends a text-to-speech request to the provided Gradio interface URL.

    Args:
        interface_url (str): URL of the Gradio interface.
        text (str): Text to convert to speech.
        text_language (str): Language code of the text.
        refer_wav_path (str, optional): Path to a reference audio clip (optional). Defaults to "".
        prompt_text (str, optional): Optional prompt text (optional). Defaults to "".
        prompt_language (str, optional): Language code for the prompt text (optional). Defaults to "".

    Returns:
        dict or bytes: Response from the Gradio interface. The format depends on the interface's output.
    """

    #client = Client(interface_url)

    input_data = {
    "text": text,#"But truly, is a simple piece of paper worth the credit people give it?",                   # str.(required) text to be synthesized
    "text_lang": text_lang,              # str.(required) language of the text to be synthesized
    "ref_audio_path": ref_audio_path,         # str.(required) reference audio path.
    "prompt_text": prompt_text,            # str.(optional) prompt text for the reference audio
    "prompt_lang": prompt_lang,            # str.(required) language of the prompt text for the reference audio
    "top_k": top_k,                   # int.(optional) top k sampling
    "top_p": top_p,                   # float.(optional) top p sampling
    "temperature": temperature,             # float.(optional) temperature for sampling
    "text_split_method": text_split_method,  # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": batch_size,              # int.(optional) batch size for inference
    "batch_threshold": batch_threshold,      # float.(optional) threshold for batch splitting.
    "split_bucket": split_bucket,         # bool.(optional) whether to split the batch into multiple buckets.
    "speed_factor":speed_factor,           # float.(optional) control the speed of the synthesized audio.
    "fragment_interval":fragment_interval,      # float.(optional) to control the interval of the audio fragment.
    "seed": seed,                   # int.(optional) random seed for reproducibility.
    "media_type": media_type,          # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": streaming_mode,      # bool.(optional) whether to return a streaming response.
    "parallel_infer": parallel_infer,       # bool.(optional) whether to use parallel inference.
    "repetition_penalty": repetition_penalty    # float.(optional) repetition penalty for T2S model.
}
    url = "http://127.0.0.1:9880/tts"

    print("hello from sendtts", tts_queue.qsize())
    response = requests.post(url, json=input_data) #response will be a .wav type of bytes
    tts_queue.put(response.content)  # Enqueue the audio data
    # audio_playback(response.content)
    if not playback_thread.is_alive():
        playback_thread.start()


def audio_playback(audio_data=None):

    paused = False
    def on_press(key):
        nonlocal paused  # Modify the 'paused' variable from the nested function
        if key == keyboard.Key.f3:
            paused = not paused

    # Create a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    stop_requested = False

    # Open an audio stream using pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,  # Assuming 16-bit signed integer PCM
                    channels=1,               # Assuming mono audio
                    rate=32000,               # Example framerate (replace with actual value)
                    output=True)

    # Write audio data to the stream in chunks
    while not stop_requested:
        # Initialize audio buffer with silence padding to avoid popping sound
        if paused:
            time.sleep(0.1)
            continue
        try:
            print("hello from playback", tts_queue.qsize())
            audio_data = tts_queue.get(timeout=1)  # Get audio data from the queue
            audio_data = audio_data[64:]#skip first 64 bytes to avoid popping sound
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            continue
        while audio_data:
            if not paused:
                data = audio_data[:1024]
                stream.write(data)
                audio_data = audio_data[len(data):]

        # if len(audio_data )== 0:
        #     stop_requested = True
        # # Check for stop request (implementation depends on your program)
        # # ... (e.g., user input, flag set elsewhere)

        # # Write a chunk of audio data to the stream
        # data = audio_data[:4096]  # Write a buffer of 4096 bytes (adjust based on performance)
        # stream.write(data)
        # # Remove processed data from audio_data to avoid infinite loop
        # audio_data = audio_data[len(data):]

    # Close the audio stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

playback_thread = threading.Thread(target=audio_playback, daemon=True)

# #only for testing
# if __name__ == "__main__":
#     # Example usage (replace with your Gradio interface URL)
#     response = send_tts_request()

#     # Process the response based on the Gradio interface's output format
#     # (e.g., extract audio data, handle errors)
#     print(f"Response: {response}")
#. The playback_thread does not get terminated when the main thread is terminated