
### Streaming Inference Example (No GPU)

You can stream audio without a GPU by using `orpheus-cpp`, which is a llama.cpp-compatible backend of the Orpheus TTS model.

1. Install orpheus-cpp
   ```bash
   pip install orpheus-cpp
   ```
2. Install llama-cpp-python
   #### Linux/Windows
   ```console
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   ```

   #### MacOS with Apple Silicon
   ```console
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
   ```
3. Run the example below:
   ```python
   from scipy.io.wavfile import write
   from orpheus_cpp import OrpheusCpp
   import numpy as np

   orpheus = OrpheusCpp(verbose=False, lang="en")

   text = "I really hope the project deadline doesn't get moved up again."
   buffer = []
   for i, (sr, chunk) in enumerate(orpheus.stream_tts_sync(text, options={"voice_id": "tara"})):
      buffer.append(chunk)
      print(f"Generated chunk {i}")
   buffer = np.concatenate(buffer, axis=1)
   write("output.wav", 24_000, np.concatenate(buffer))
   ```
4. WebRTC Streaming Example:
   ```bash
   python -m orpheus_cpp
   ```
   <video src="https://github.com/user-attachments/assets/54dfffc9-1981-4d12-b4d1-eb68ab27e5ad" controls style="text-align: center">></video>
