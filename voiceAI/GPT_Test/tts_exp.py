from gradio_client import Client
import json, requests
import pyaudio
from pynput import keyboard
from queue import Queue
import threading
import time
import sys
import asyncio
sys.path.append('voiceAI/GPT_Test/')
tts_queue = Queue()
# Lock for thread-safe operations
LOCK = threading.Lock()
CONDITION = threading.Condition(LOCK)
PLAYBACK_PAUSED = threading.Event()

import queue

audio_queue = queue.Queue(maxsize=10)  # Buffer for audio chunks

# def audio_playback():
#     def on_press(key):
#         if key == keyboard.Key.f3:
#             if PLAYBACK_PAUSED.is_set():
#                 PLAYBACK_PAUSED.clear()
#                 print("Playback resumed")
#             else:
#                 PLAYBACK_PAUSED.set()
#                 print("Playback paused")

#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()

#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=32000,
#                     output=True)

#     def audio_worker():
#         while True:
#             if not PLAYBACK_PAUSED.is_set():
#                 try:
#                     chunk = audio_queue.get(timeout=0.1)
#                     stream.write(chunk)
#                 except queue.Empty:
#                     time.sleep(0.01)  # Short sleep if queue is empty
#             else:
#                 time.sleep(0.1)  # Longer sleep when paused

#     worker_thread = threading.Thread(target=audio_worker)
#     worker_thread.daemon = True
#     worker_thread.start()

#     while True:
#         with CONDITION:
#             while tts_queue.empty():
#                 CONDITION.wait()
#             try:
#                 audio_data = tts_queue.get(timeout=1)
#                 audio_data = audio_data[64:]  # skip first 64 bytes
#                 while audio_data:
#                     chunk = audio_data[:2048]
#                     audio_queue.put(chunk)
#                     audio_data = audio_data[2048:]
#             except Exception as e:
#                 print(f"Error: {e}")
#     listener.stop()
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

def audio_playback(audio_data=None):
    def on_press(key):
        if key == keyboard.Key.f3:
            if PLAYBACK_PAUSED.is_set():
                PLAYBACK_PAUSED.clear()
            else:
                PLAYBACK_PAUSED.set()
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
        with CONDITION:
            while tts_queue.empty():
                CONDITION.wait()
            # Initialize audio buffer with silence padding to avoid popping sound
            try:
                audio_data = tts_queue.get(timeout=1)  # Get audio data from the queue
                audio_data = audio_data[64:]#skip first 64 bytes to avoid popping sound
                while audio_data:
                    if PLAYBACK_PAUSED.is_set():
                        time.sleep(0.1)
                        continue
                    data = audio_data[:2048]
                    stream.write(data)
                    audio_data = audio_data[len(data):]
            except Exception as e:
                print(f"Error: {e}")
                continue
    # Close the audio stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
#prompt_text example, causes issues: But truly, is a simple piece of paper worth the credit people give it?
async def send_tts_request(text="(Super Elite Magnificent Agent John Smith!)", text_lang="en",
                        ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav",
                          prompt_text="", prompt_lang="en",
                          top_k=7, top_p=.87, temperature=0.87,
                          text_split_method="cut5",
                          batch_size=1, batch_threshold=0.45, split_bucket=True,
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

    input_data = {
    "text": text,#"""So as a joke, I went to my friend's house wearing Pekora's wig and clothes. I could barely stop my laughter as he went as red as a tomato and looked at me from head to toe with a bit of drool in his mouth. The way he stared made mde feel a bit funny too, but I decided to tease him more by taking off my clothes. He asked me, 'Are you serious?' and I said 'Yep peko.' He went silent for what seemed like forever, so I asked him, 'What's the matter peko?' He said he's confused, but then his boner got really hard, which made me take off his clothes. I expected him to scream, 'Stop!' as I kissed him and stroked his cock, but he instead shouted 'Oh God, Pekora!' which made me get a boner myself. Before I knew it, I was blowing him for the first time till he came. His semen was so thick, it got stuck inside my throat no matter how hard I swallowed. He then said, 'I want to fuck you now!' and seeing that we've already gone that far and we were both naked, I obliged. A few hours later, the jerk went all pale and said to me 'Why did we do that? Now I'm not fucking straight.' But he still looked so cute all confused like that, so I took pity on him and reassured while wiping his cum off my face, 'Let's just pretend I'ms till Pekora""",
    #"Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew Hello very long text needs more or less text i may have made a serious error and a mistake that I ened to solve every one ce ilew",#"""So as a joke I went to my friend's house wearing Pekora's wig and clothes I could barely stop my laughter as he went as red as a tomato and looked at me from head to toe with a bit of drool in his mouth The way he stared made mde feel a bit funny too but I decided to tease him more by taking off my clothes He asked me 'Are you serious' and I said 'Yep peko' He went silent for what seemed like forever so I asked him 'What's the matter peko' He said he's confused but then his boner got really hard which made me take off his clothes I expected him to scream 'Stop' as I kissed him and stroked his cock but he instead shouted 'Oh God Pekora' which made me get a boner myself Before I knew it I was blowing him for the first time till he came His semen was so thick it got stuck inside my throat no matter how hard I swallowed He then said 'I want to fuck you now' and seeing that we've already gone that far and we were both naked I obliged A few hours later the jerk went all pale and said to me 'Why did we do that Now I'm not fucking straight' But he still looked so cute all confused like that so I took pity on him and reassured while wiping his cum off my face 'Let's just pretend I'm still Pekora' """, #"I think whats going on is the file is being generated, then translated to bytes, then streamed by fastapi  I think whats going on is the file is being generated, then translated to bytes, then streamed by fastapi I think whats going on is the file is being generated, then translated to bytes, then streamed by fastapi But truly, is a simple piece of paper worth the credit people give it?",                   # str.(required) text to be synthesized
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
    "return_fragment":True,      # float.(optional) to control the interval of the audio fragment.
    "seed": seed,                   # int.(optional) random seed for reproducibility.
    "media_type": media_type,          # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": streaming_mode,      # bool.(optional) whether to return a streaming response.
    "parallel_infer": parallel_infer,       # bool.(optional) whether to use parallel inference.
    "repetition_penalty": repetition_penalty    # float.(optional) repetition penalty for T2S model.
}
    url = "http://127.0.0.1:9880/tts"
    # for data in response.content:
    #     print(data)
    # s = time.perf_counter()

    # response = requests.post(url, json=input_data) #response will be a .wav type of bytes
    # e = time.perf_counter()
    # print("SJSJSJSJSJSJSJS", e-s)
    # with LOCK:
    #     tts_queue.put(response.content)  # Enqueue the audio data
    #     CONDITION.notify()
    from api_v2 import tts_direct

    # resp = tts_direct(input_data)
    # s = time.perf_counter()
    # for r in resp:
    #     with LOCK:
    #         if tts_queue.empty:
    #             CONDITION.notify()
    #         tts_queue.put(r)  # Enqueue the audio data
    #     e = time.perf_counter()
    #     print("TIME TAKEN", e-s)
    #     # audio_playback_v2(r)
    #     # print("_-----"*30,r[:5], '\n')
    #     s = time.perf_counter()
    # print(resp)
    resp = tts_direct(input_data)
    s = time.perf_counter()
    for r in resp:
        await asyncio.to_thread(enqueue_audio, r)
        e = time.perf_counter()
        print("TIME TAKEN", e-s)
        s = time.perf_counter()

def enqueue_audio(audio_data):
    with LOCK:
        if tts_queue.empty():
            CONDITION.notify()
        tts_queue.put(audio_data)


playback_thread = threading.Thread(target=audio_playback, daemon=True)
playback_thread.start()


if __name__ == "__main__":
    import asyncio
    while True:
        asyncio.run(send_tts_request())
        time.sleep(15)

    # while True:
    #     asyncio.run(send_tts_request())
        # time.sleep(5000)
    #windows venv implementation
    # VENV = 'venv'
    # import os
    # import subprocess
    # def run_TTS():
    #     venv_path = f"..\\..\\{VENV}\\"
    #     script1_path = "./tts_exp.py"
    #     new_dir = "./voiceAI/GPT_Test"
    #     # Activate virtual environment and run the script
    #     activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
    #     command = f"{activate_script} && python {script1_path}"
    #     try:
    #         subprocess.run(command, shell=True, check=True, cwd=new_dir)
    #     except subprocess.CalledProcessError as e:
    #         print("Error running TTS:", e)
    # run_TTS()

#######only for testing, requires GPT-SoVITTS to be up and running, WIP, needs to be fixed up
# if __name__ == "__main__":
#     import subprocess
#     import os
#     print(os.getcwd())
#     script1_path = "./api_v2.py"
#     # venv_path = "venv2\\scripts\\activate"
#     venv_path = "..\\..\\venv2\\"
#     def is_server_ready(url):#abuses the fact that an unready server won't work at
#         try:
#             requests.get(url, timeout=5)  # Set a timeout for the health check request
#             print("Server up and running")
#             return True
#         except requests.exceptions.RequestException as e:
#             print(f"Error checking server health: {e}")
#             return False
#     def run_TTS():
#         new_dir = "./voiceAI/GPT-SoVITS-fast_inference/"
#         # Activate virtual environment and run the script in background
#         activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
#         command = f"{activate_script} && python {script1_path}"
#         subprocess.Popen(command, shell=True, cwd=new_dir)
#         print("TTS server started in background")
#         server_health_check_url = "http://localhost:9880"  # Replace with your actual URL
#         while not is_server_ready(server_health_check_url):
#             print("Waiting for TTS server to be ready...")
#             time.sleep(2)
#     run_TTS()
#     response = send_tts_request()
#     while True:
#         continue