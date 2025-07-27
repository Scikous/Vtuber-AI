import aiohttp
import asyncio
import time
# API_BASE_URL = "http://127.0.0.1:9880"  # Default, should be configurable if needed
API_BASE_URL = "http://0.0.0.0:9880"  # Default, should be configurable if needed

async def send_tts_request(text: str,
                           text_lang: str,
                           ref_audio_path: str,
                           prompt_lang: str,
                           prompt_text: str = "",
                           top_k: int = 7,
                           top_p: float = 0.87,
                           temperature: float = 0.87,
                           text_split_method: str = "cut5",
                           speed_factor: float = 1.0,
                           media_type: str = "wav",
                           streaming_mode: bool = True, # Crucial for streaming
                           # Add other relevant parameters from TTS_Request model in api_v2.py
                           aux_ref_audio_paths: list = None,
                           batch_size: int = 1,
                           batch_threshold: float = 0.45,
                           split_bucket: bool = True,
                           fragment_interval: float = 0.3,
                           seed: int = -1,
                           parallel_infer: bool = True,
                           repetition_penalty: float = 1.35,
                           sample_steps: int = 32,
                           super_sampling: bool = False,
                           logger=None):
    """
    Sends a request to the TTS API (api_v2.py) and streams the audio response.
    Yields audio chunks.
    """
    endpoint = f"{API_BASE_URL}/tts"
    params = {
        "text": text,
        "text_lang": text_lang,
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "speed_factor": speed_factor,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "aux_ref_audio_paths": aux_ref_audio_paths if aux_ref_audio_paths else [],
        "batch_size": batch_size,
        "batch_threshold": batch_threshold,
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": sample_steps,
        "super_sampling": super_sampling
    }

    # Filter out None values from params, as FastAPI handles defaults
    params = {k: v for k, v in params.items() if v is not None}

    # Convert boolean values to strings for query parameters
    for key, value in params.items():
        if isinstance(value, bool):
            params[key] = str(value).lower()

    if logger:
        logger.debug(f"TTS Request to {endpoint} with params: {text[:50]}..., lang:{text_lang}, streaming:{streaming_mode}")
    start = time.perf_counter()
    session = aiohttp.ClientSession()
    try:
        async with session.get(endpoint, params=params) as response:
                if response.status == 200:
                    print("TTS Request Text:", text)
                    if streaming_mode:
                        async for chunk in response.content.iter_any(): # iter_any for immediate chunks
                            if chunk: # Ensure chunk is not empty
                                yield chunk[64:]
                    else:
                        # For non-streaming, read the whole content, though this path isn't expected for this use case
                        audio_data = await response.read()
                        end = time.perf_counter()
                        print("AUDIO GEN TIME)"*10, end-start)
                        yield audio_data[64:]
                else:
                    error_content = await response.text()
                    if logger:
                        logger.error(f"TTS API request failed with status {response.status}: {error_content}")
                    # Optionally, raise an exception or yield an error indicator
                    # For now, just log and yield nothing further on error
                    pass # Or raise Exception(f"TTS API Error: {response.status} - {error_content}")
    except aiohttp.ClientConnectorError as e:
        if logger:
            logger.error(f"TTS API connection error: {e}")
        # Optionally, raise an exception or yield an error indicator
        pass # Or raise Exception(f"TTS API Connection Error: {e}")
    finally:
        if session: # Check if session was initialized
            await session.close()

# # Example usage (for testing tts_exp.py directly)
# async def main_test():
#     # This is a placeholder for actual paths and parameters
#     # You'll need a running api_v2.py instance
#     # And valid reference audio paths accessible by api_v2.py
#     print("Testing TTS streaming... Ensure api_v2.py is running.")
#     # Example, replace with actual valid paths for your setup
#     # ref_audio_path_example = "path/to/your/reference_audio.wav"
#     # if not os.path.exists(ref_audio_path_example):
#     #     print(f"Warning: Example ref_audio_path '{ref_audio_path_example}' does not exist. TTS might fail.")

#     # Create a dummy logger for testing
#     class DummyLogger:
#         def debug(self, msg): print(f"DEBUG: {msg}")
#         def info(self, msg): print(f"INFO: {msg}")
#         def error(self, msg): print(f"ERROR: {msg}")

#     logger = DummyLogger()

#     try:
#         async for audio_chunk in send_tts_request(
#             text="Hello world, this is a streaming test.",
#             text_lang="en",
#             # Replace with a valid path accessible by the api_v2.py server
#             ref_audio_path="../dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav", 
#             prompt_text="This is a prompt.",
#             prompt_lang="en",
#             streaming_mode=True,
#             media_type='wav',
#             logger=logger
#         ):
#             logger.info(f"Received audio chunk of size: {len(audio_chunk)}")
#             # In a real application, you'd play this chunk or save it
#     except Exception as e:
#         logger.error(f"Error during TTS test: {e}")

# if __name__ == "__main__":
#     # Note: To run this test, you need api_v2.py running and aiohttp installed.
#     # You might also need to adjust the ref_audio_path to something valid for your api_v2.py server.
#     # Example: python -m TTS_Wizard.GPT_SoVITS.tts_exp (if in the parent directory and paths are set up)
#     # Or simply: python tts_exp.py (if running from TTS_Wizard/GPT_SoVITS/)
#     asyncio.run(main_test())






