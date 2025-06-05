import os
import subprocess
import sys
import signal # For specific signals if needed, and for CTRL_C_EVENT on Windows

# Define the virtual environment names. This could be read from a config or .env file.
VENV_NAME_AI = 'venvRun'
VENV_NAME_TTS = 'venvTTS'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PATH = '/home/santa/' # Make sure this is correct for your Linux setup


# def get_venv_environment(venv_name):
#     """Get the environment variables for a virtual environment."""
#     venv_path = os.path.join(VENV_PATH, venv_name)
#     env = os.environ.copy()  # Copy the current environment
#     if sys.platform != "win32":
#         # Update PATH to include the virtual environment's bin directory
#         env["PATH"] = f"{os.path.join(venv_path, 'bin')}:{env.get('PATH', '')}"
#         # Update LD_LIBRARY_PATH to include CUDA/cuDNN libraries if needed
#         cuda_lib_path = "/usr/local/cuda/lib64"  # Adjust based on your CUDA installation
#         cudnn_lib_path = "/usr/lib/x86_64-linux-gnu"  # Adjust based on your cuDNN installation
#         env["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{cudnn_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
#         env["CUDA_HOME"] = "/usr/local/cuda"  # Adjust if CUDA is installed elsewhere
#     return env


def get_python_executable(venv_name):
    """Gets the path to the Python executable in the specified virtual environment."""
    print(f"Attempting to find Python for venv: {venv_name}")
    if sys.platform == "win32":
        # On Windows, venvs are often directly in the project root
        python_exe = os.path.join(PROJECT_ROOT, venv_name, "Scripts", "python.exe")
        # Or, if you have a global venv directory like VENV_PATH
        # python_exe = os.path.join(VENV_PATH, venv_name, "Scripts", "python.exe")
    else:  # Linux/macOS
        python_exe = os.path.join(VENV_PATH, venv_name, "bin", "python")

    print(f"Checking path: {python_exe}")
    if os.path.exists(python_exe):
        print(f"Found: {python_exe}")
        return python_exe
    print(f"Not found: {python_exe}")
    return None

def main():
    """
    Main function to start the Vtuber-AI application by running the MainOrchestrator,
    and the TTS service by running the api_v2.py script.
    """
    print("Starting Vtuber-AI application and TTS service...")

    # Prepare the command for the Vtuber-AI main orchestrator
    python_executable_ai = get_python_executable(VENV_NAME_AI)
    main_orchestrator_script = os.path.join(PROJECT_ROOT, "src", "main_orchestrator.py")

    if not os.path.exists(main_orchestrator_script):
        print(f"Error: Main orchestrator script not found at {main_orchestrator_script}")
        sys.exit(1)

    if python_executable_ai:
        print(f"Using Python from virtual environment: {python_executable_ai}")
        command_ai = [python_executable_ai, main_orchestrator_script]
    else:
        print(f"Warning: Virtual environment '{VENV_NAME_AI}' Python not found. "
              f"Attempting to use system 'python'.")
        print("Please ensure the virtual environment is activated and contains all dependencies, "
              "or that dependencies are installed globally.")
        command_ai = ["python", main_orchestrator_script]

    # Prepare the command for the TTS service
    python_executable_tts = get_python_executable(VENV_NAME_TTS)
    tts_script = os.path.join(PROJECT_ROOT, "TTS_Wizard", "GPT_SoVITS", "api_v2.py")
    # Corrected path for tts_config, assuming it's inside the second GPT_SoVITS
    tts_config = os.path.join(PROJECT_ROOT, "TTS_Wizard", "GPT_SoVITS", "GPT_SoVITS", "configs", "tts_infer.yaml")


    if not os.path.exists(tts_script):
        print(f"Error: TTS script not found at {tts_script}")
        sys.exit(1)
    if not os.path.exists(tts_config):
        print(f"Error: TTS config not found at {tts_config}")
        sys.exit(1)

    if python_executable_tts:
        print(f"Using Python from virtual environment: {python_executable_tts}")
        command_tts = [python_executable_tts, tts_script, "-a", "0.0.0.0", "-p", "9880", "-c", tts_config]
    else:
        print(f"Warning: Virtual environment '{VENV_NAME_TTS}' Python not found. "
              f"Attempting to use system 'python'.")
        print("Please ensure the virtual environment is activated and contains all dependencies, "
              "or that dependencies are installed globally.")
        command_tts = ["python", tts_script, "-a", "0.0.0.0", "-p", "9880", "-c", tts_config]

    print(f"Executing AI: {' '.join(command_ai)}")
    print(f"Executing TTS: {' '.join(command_tts)}")
    print(f"Project Root: {PROJECT_ROOT}")

    process_ai = None
    process_tts = None

    try:
        # # Get environment for each virtual environment
        # env_ai = get_venv_environment(VENV_NAME_AI)
        # env_tts = get_venv_environment(VENV_NAME_TTS)

        # For Windows, CREATE_NEW_PROCESS_GROUP allows sending CTRL_BREAK_EVENT to the group.
        # For Unix, preexec_fn=os.setsid creates a new session and process group.
        # This helps in ensuring signals are delivered correctly to the children and their descendants.
        common_popen_kwargs = {}
        if sys.platform == "win32":
            common_popen_kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else: # Linux/macOS
            common_popen_kwargs['preexec_fn'] = os.setsid

        # Run the main_orchestrator.py script
        process_ai = subprocess.Popen(command_ai, cwd=PROJECT_ROOT, **common_popen_kwargs)
        # process_ai = subprocess.Popen(command_ai, cwd=PROJECT_ROOT, env=env_ai, **common_popen_kwargs)

        print(f"Started AI process with PID: {process_ai.pid}")

        # Run the TTS service in the TTS_Wizard/GPT_SoVITS directory
        tts_cwd = os.path.join(PROJECT_ROOT, "TTS_Wizard", "GPT_SoVITS")
        process_tts = subprocess.Popen(command_tts, cwd=tts_cwd, **common_popen_kwargs)
        # process_tts = subprocess.Popen(command_tts, cwd=tts_cwd, env=env_tts, **common_popen_kwargs)

        print(f"Started TTS process with PID: {process_tts.pid}")

        # Wait for both processes to complete
        # If one finishes or crashes, the other wait() will still be called.
        if process_ai:
            process_ai.wait()
        if process_tts:
            process_tts.wait()

    except FileNotFoundError as e:
        # This specific error usually means the executable (python or script) wasn't found
        print(f"Error: Failed to execute command. Ensure executable is valid and in PATH.")
        print(f"Details: {e}")
        # No need to sys.exit(1) here, finally block will still run.
    except subprocess.CalledProcessError as e: # This is for check_call or check_output, not Popen directly
        print(f"Error running process: {e}")
    except KeyboardInterrupt:
        print("\nApplication interrupted by user (Ctrl+C in run.py). Terminating child processes...")
        # The finally block will handle the actual termination.
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Cleaning up processes...")
        processes_to_handle = []
        if process_ai:
            processes_to_handle.append(("AI", process_ai))
        if process_tts:
            processes_to_handle.append(("TTS", process_tts))

        for name, proc in processes_to_handle:
            if proc.poll() is None:  # Check if process is still running
                print(f"Attempting to terminate {name} process (PID: {proc.pid})...")
                try:
                    if sys.platform == "win32":
                        # Send CTRL_BREAK_EVENT to the process group on Windows
                        # This is often more effective for console apps than terminate()
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        # On Unix, send SIGTERM to the entire process group
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Could not send signal to {name} (PID: {proc.pid}): {e}. Falling back to terminate().")
                    proc.terminate() # Fallback or primary for non-group scenarios

                try:
                    proc.wait(timeout=5)  # Wait for graceful termination
                    print(f"{name} process (PID: {proc.pid}) terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print(f"{name} process (PID: {proc.pid}) did not terminate gracefully after 5s, killing...")
                    # On Unix, kill the process group
                    if sys.platform != "win32":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception as e_killpg: # Process might have died in the meantime
                             print(f"Could not SIGKILL process group for {name} (PID: {proc.pid}): {e_killpg}")
                             proc.kill() # Fallback to killing only the main child process
                    else:
                        proc.kill() # SIGKILL equivalent on Windows
                    proc.wait() # Ensure it's dead
                    print(f"{name} process (PID: {proc.pid}) killed.")
                except Exception as e_wait: # Catch other errors during wait
                    print(f"Error waiting for {name} process (PID: {proc.pid}) to terminate: {e_wait}")
            else:
                print(f"{name} process (PID: {proc.pid}) already finished (return code: {proc.returncode}).")

        print("Vtuber-AI application and TTS service cleanup finished.")

if __name__ == "__main__":
    main()