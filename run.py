import os
from threading import Thread
import subprocess
# Define the paths to your Python files
script2_path = "brain.py"
script1_path = "api_v2.py"


def run_TTS():
    new_dir = "./voiceAI/GPT-SoVITS-fast_inference/"
    try:
        subprocess.run(["python", "api_v2.py"], check=True,
                       cwd=new_dir)  # Run script1 with arguments
    except subprocess.CalledProcessError as e:
        print("Error running TTS:", e)


def run_brain():
    os.system(f"python {script2_path}")

if __name__ == "__main__":
    tts_thread = Thread(target=run_brain, daemon=True)
    tts_thread.start()
    run_TTS()
