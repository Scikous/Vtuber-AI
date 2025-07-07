# import sys
# from pathlib import Path

# # Assuming tts_exp.py is in a directory like 'TTS_Wizard',
# # and 'Coqui_TTS' is a subdirectory within it.
# # We need to add the 'Coqui_TTS' directory itself to the Python path.
# # This allows the library's internal 'from TTS.utils...' imports to work correctly.
# try:
#     # This determines the path to the directory containing this script.
#     script_dir = Path(__file__).resolve().parent
#     # This constructs the path to the Coqui_TTS library directory.
#     # IMPORTANT: Adjust this if your tts_exp.py is not in the 'TTS_Wizard' parent directory of Coqui_TTS.
#     # For example, if tts_exp.py is at the project root, this might be: Path('TTS_Wizard/Coqui_TTS')
#     coqui_tts_path = script_dir / 'Coqui_TTS'
    
#     if not coqui_tts_path.exists():
#         raise FileNotFoundError(f"Coqui_TTS directory not found at expected path: {coqui_tts_path}")

#     # Add the Coqui_TTS directory to the system path.
#     # We use insert(0, ...) to give it priority over other installed packages.
#     sys.path.insert(0, str(coqui_tts_path))
    
# except NameError:
#     # If __file__ is not defined (e.g., in an interactive session), use a relative path.
#     # This is less robust but can work for simple cases.
#     print("Warning: __file__ not defined. Assuming 'Coqui_TTS' is in the current working directory.")
#     sys.path.insert(0, 'Coqui_TTS')
# # --- End of Path Correction ---




import torch
from pathlib import Path
import time
import logging
import uvicorn
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- IMPORTANT: Fix relative imports for script execution ---
# When running a script directly, relative imports like `from .Coqui_TTS...` can fail.
# We'll adjust the path to make the imports absolute based on the script's location.
# This assumes a directory structure like:
# PROJECT_ROOT/
# ├── TTS_Wizard/
# |   ├── tts_exp.py
# |   └── Coqui_TTS/
# |   └── utils/
# └── run.py

# # import sys
# # Get the directory of the current script (tts_exp.py)
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # Get the parent directory (TTS_Wizard)
# tts_wizard_dir = os.path.dirname(script_dir)
# # Add the project root to the Python path to make top-level imports work
# project_root_dir = os.path.dirname(tts_wizard_dir) # Adjust if your structure is different
# sys.path.append(project_root_dir)

# Now these should work as top-level imports
from TTS_Wizard.Coqui_TTS.TTS.utils.manage import ModelManager
from TTS_Wizard.Coqui_TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS_Wizard.Coqui_TTS.TTS.tts.models.xtts import Xtts

# --- Configuration ---
# You can move these to environment variables or a config file for better practice
SPEAKER_WAV_PATH = "TTS_Wizard/dataset/inference_testing/vocal_john10.wav.reformatted.wav_10.wav"
SERVER_HOST = "0.0.0.0"  # Listen on all network interfaces
SERVER_PORT = 8002       # Port for the TTS service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Model for Request Body ---
class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speech_speed: float = 1.0
    # Add any other XTTS params you want to control
    temperature: float = 0.75
    repetition_penalty: float = 1.0

#--- XTTS Service Class (largely unchanged) ---
class XTTS_Service:
    """
    A service class for Text-to-Speech using Coqui's XTTS model.
    It handles model loading, setup, and provides a method for streaming TTS inference.
    """
    def __init__(self, speaker_wav_path: str, device: str = "auto"):
        """
        Initializes the XTTS service. This is a heavy operation as it loads the model into memory.

        Args:
            speaker_wav_path (str): The path to the speaker reference audio file.
            device (str, optional): The device to run the model on ('cuda', 'cpu', or 'auto'). Defaults to "auto".
        """
        self.device = self._determine_device(device)
        logger.info(f"Using device: {self.device}")

        # --- 1. Load Model and Config ---
        self.config, self.model = self._load_model()
        
        # --- 2. Get Speaker Conditioning Latents ---
        if not Path(speaker_wav_path).exists():
            raise FileNotFoundError(f"Speaker reference audio file not found: {speaker_wav_path}")
        
        logger.info(f"Computing speaker conditioning latents from: {speaker_wav_path}")
        self.gpt_cond_latents, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_wav_path,
            gpt_cond_len=self.config.model_args.get("gpt_cond_len", 30),
            gpt_cond_chunk_len=self.config.model_args.get("gpt_cond_chunk_len", 4),
            max_ref_length=self.config.model_args.get("max_ref_len", 30),
            sound_norm_refs=self.config.model_args.get("sound_norm_refs", False)
        )
        logger.info("Speaker conditioning latents computed. XTTS_Service is ready.")

    def _determine_device(self, device_str: str) -> str:
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_str

    def _load_model(self):
        """
        Downloads (if necessary) and loads the XTTSv2 model.
        """
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Loading model: {model_name}...")
        
        try:
            from .Coqui_TTS.TTS.api import TTS as ApiTTS_for_path
            models_file_path = ApiTTS_for_path.get_models_file_path()
        except ImportError:
            logger.warning("Could not determine .models.json path via TTS.api.")
            models_file_path = None
            
        manager = ModelManager(models_file=models_file_path, progress_bar=True)
        
        model_path, config_path, _ = manager.download_model(model_name)
        
        # Load configuration
        config = XttsConfig()
        config.load_json(config_path)

        # Initialize model
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        model.to(self.device)
        
        logger.info("XTTS model loaded successfully.")
        return config, model


    def send_tts_request(self, text: str, language: str, **kwargs):
        stream_params = {
            "stream_chunk_size": self.config.model_args.get("stream_chunk_size", 20),
            "overlap_wav_len": self.config.model_args.get("overlap_wav_len", 1024),
            "length_penalty": self.config.length_penalty,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "repetition_penalty": kwargs.get('repetition_penalty', self.config.repetition_penalty),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "speech_speed": kwargs.get('speech_speed', 1.0),
            "enable_text_splitting": True,
        }

        logger.info(f"Starting streaming inference for text: '{text[:50]}...'")
        stream = self.model.inference_stream(
            text, language, self.gpt_cond_latents, self.speaker_embedding, **stream_params
        )
        for chunk in stream:
            logger.info(f"Yielded audio chunk")

            yield chunk.cpu().numpy().tobytes()

if __name__ == "__main__":

    TTS = XTTS_Service(speaker_wav_path=SPEAKER_WAV_PATH)
    
    import time
    start = time.perf_counter()
    print("Started tts request")
    job = TTS.send_tts_request(text="Hello World lmao, so great and fantastic it is. Wowzer, womp womp", language="en")
    li = []
    for res in job:
        li.append(res)
        end = time.perf_counter()
        print(end-start)





    # import torch
    # from TTS_Wizard.Coqui_TTS.TTS.api import TTS

    # # Get device
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # List available 🐸TTS models
    # # print(TTS().list_models())

    # # Initialize TTS
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # # List speakers
    # # print(tts.speakers)

    # # Run TTS
    # # ❗ XTTS supports both, but many models allow only one of the `speaker` and
    # # `speaker_wav` arguments
    # start = time.perf_counter()
    # # TTS with list of amplitude values as output, clone the voice from `speaker_wav`
    # wav = tts.tts(
    # text="world!",
    # speaker_wav=SPEAKER_WAV_PATH,
    # language="en"
    # )
    # end = time.perf_counter()

    print(end-start)


    # # TTS to a file, use a preset speaker
    # tts.tts_to_file(
    # text="Hello world!",
    # speaker="Craig Gutsy",
    # language="en",
    # file_path="output.wav"
    # )




        
    # # --- FastAPI App Setup ---
    # app = FastAPI()

    # # Load the model on startup. This is a heavy operation and should only happen once.
    # try:
    #     logger.info("Initializing XTTS Service...")
    #     tts_service = XTTS_Service(speaker_wav_path=SPEAKER_WAV_PATH)
    #     logger.info("XTTS Service initialized successfully.")
    # except Exception as e:
    #     logger.error(f"Failed to initialize XTTS_Service: {e}", exc_info=True)
    #     tts_service = None # Ensure tts_service is defined

    # @app.post("/tts")
    # async def generate_speech(request: TTSRequest):
    #     """
    #     API endpoint to generate speech. It streams the audio back to the client.
    #     """
    #     if not tts_service:
    #         return {"error": "TTS service is not available."}, 503

    #     def stream_generator():
    #         # This generator calls the main TTS generator and yields its chunks
    #         yield from tts_service.send_tts_request(
    #             text=request.text,
    #             language=request.language,
    #             speech_speed=request.speech_speed,
    #             temperature=request.temperature,
    #             repetition_penalty=request.repetition_penalty
    #         )

    #     # Return a streaming response, which FastAPI handles efficiently
    #     return StreamingResponse(stream_generator(), media_type="application/octet-stream")

    # @app.get("/health")
    # async def health_check():
    #     """Health check endpoint to verify the service is running."""
    #     return {"status": "ok" if tts_service else "error", "service": "XTTS_v2"}
    # # This block now starts the web server
    # logger.info(f"Starting TTS server on http://{SERVER_HOST}:{SERVER_PORT}")
    # uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)