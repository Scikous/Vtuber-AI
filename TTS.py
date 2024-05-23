
from gradio_client import Client
import json, requests
import sounddevice as sd
import winsound
import pyaudio
from pynput import keyboard

def send_tts_request(text="Super Elite Magnificent Agent John Smith!", text_lang="en",
                        ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav",
                          prompt_text="But truly, is a simple piece of paper worth the credit people give it?", prompt_lang="en",
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
#"&batch_size=1&media_type=wav&streaming_mode=true&top_k=5&top_p=1&temperature=1"
    #response = client.predict(refer_wav_path, api_name="/tts")
    #response = requests.post(interface_url, json=input_data)
    #print(response.status_code)
    url = "http://127.0.0.1:9880/tts"

    response = requests.post(url, json=input_data) #response will be a .wav type of bytes
    # print(type(response.content))
    # for cont in response.content:
    #     print(cont)
    #     winsound.PlaySound(cont, winsound.SND_MEMORY) 
    #     break
    #winsound.PlaySound(response.content, winsound.SND_MEMORY) 

    # with open("audio_response/output.wav", "wb") as f:
    #     f.write( response.content)
    audio_playback(response.content)


def audio_playback(audio_data):
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
        if paused:
            continue
        if len(audio_data )== 0:
            stop_requested = True
        # Check for stop request (implementation depends on your program)
        # ... (e.g., user input, flag set elsewhere)

        # Write a chunk of audio data to the stream
        data = audio_data[:4096]  # Write a buffer of 4096 bytes (adjust based on performance)
        stream.write(data)
        # Remove processed data from audio_data to avoid infinite loop
        audio_data = audio_data[len(data):]

    # Close the audio stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

    
if __name__ == "__main__":
    # Example usage (replace with your Gradio interface URL)
    response = send_tts_request()

    # Process the response based on the Gradio interface's output format
    # (e.g., extract audio data, handle errors)
    print(f"Response: {response}")
