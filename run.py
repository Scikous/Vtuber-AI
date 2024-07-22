import os
from threading import Thread
import subprocess
# Define the paths to your Python files
# venv_path = "venv2\\scripts\\activate"
VENV = 'venv'

#windows venv implementation
def run_TTS():
    venv_path = f"..\\..\\{VENV}\\"
    script1_path = "./api_v2.py"
    # script1_path = "./tts_exp.py"
    new_dir = "./voiceAI/GPT-SoVITS-fast_inference/"
    # new_dir = "./voiceAI/GPT_Test/"
    # Activate virtual environment and run the script
    activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
    command = f"{activate_script} && python {script1_path}"
    try:
        subprocess.run(command, shell=True, check=True, cwd=new_dir)
    except subprocess.CalledProcessError as e:
        print("Error running TTS:", e)

def run_brain():
    script2_path = "./brain.py"
    venv_path = f".\\{VENV}\\"
    # Activate virtual environment and run the script
    activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
    command = f"{activate_script} && python {script2_path}"
    os.system(command)

if __name__ == "__main__":
    tts_thread = Thread(target=run_brain, daemon=True)
    tts_thread.start()
    run_TTS()



#linux venv version??? untested
# def run_TTS():
#     new_dir = "./voiceAI/GPT-SoVITS-fast_inference/"
#     # Activate virtual environment and run the script
#     activate_script = os.path.join(venv_path, 'bin', 'activate')
#     command = f"source {activate_script} && python {script1_path}"
#     try:
#         subprocess.run(command, shell=True, check=True, cwd=new_dir, executable='/bin/bash')
#     except subprocess.CalledProcessError as e:
#         print("Error running TTS:", e)

# def run_brain():
#     # Activate virtual environment and run the script
#     activate_script = os.path.join(venv_path, 'bin', 'activate')
#     command = f"source {activate_script} && python {script2_path}"
#     os.system(command)

# if __name__ == "__main__":
#     tts_thread = Thread(target=run_brain, daemon=True)
#     tts_thread.start()
#     run_TTS()



#non-venv, not recommended
# print(os.getcwd())
# def run_TTS():
#     new_dir = "./voiceAI/GPT-SoVITS-fast_inference/"
#     try:
#         subprocess.run([venv_path,"python", "api_v2.py"], check=True,
#                        cwd=new_dir)  # Run script1 with arguments
#     except subprocess.CalledProcessError as e:
#         print("Error running TTS:", e)


# def run_brain():
#     os.system(f"{venv_path} python {script2_path}")

# if __name__ == "__main__":
#     tts_thread = Thread(target=run_brain, daemon=True)
#     tts_thread.start()
#     run_TTS()
