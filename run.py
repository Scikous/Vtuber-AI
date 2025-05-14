import os
import subprocess
import sys
import t
# Define the virtual environment name. This could be read from a config or .env file.
VENV_NAME = 'venv7'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_python_executable(venv_name):
    """Gets the path to the Python executable in the specified virtual environment."""
    # Construct path to venv based on OS
    if sys.platform == "win32":
        python_exe = os.path.join(PROJECT_ROOT, venv_name, "Scripts", "python.exe")
    else: # Linux/macOS
        python_exe = os.path.join(PROJECT_ROOT, venv_name, "bin", "python")
    
    if os.path.exists(python_exe):
        return python_exe
    return None

def main():
    """
    Main function to start the Vtuber-AI application by running the MainOrchestrator.
    """
    print("Starting Vtuber-AI application...")

    python_executable = get_python_executable(VENV_NAME)
    
    main_orchestrator_script = os.path.join(PROJECT_ROOT, "src", "main_orchestrator.py")

    if not os.path.exists(main_orchestrator_script):
        print(f"Error: Main orchestrator script not found at {main_orchestrator_script}")
        sys.exit(1)

    if python_executable:
        print(f"Using Python from virtual environment: {python_executable}")
        command = [python_executable, main_orchestrator_script]
    else:
        print(f"Warning: Virtual environment '{VENV_NAME}' Python not found. "
              f"Attempting to use system 'python'.")
        print("Please ensure the virtual environment is activated and contains all dependencies, "
              "or that dependencies are installed globally.")
        command = ["python", main_orchestrator_script]

    print(f"Executing: {' '.join(command)}")
    print(f"Project Root: {PROJECT_ROOT}")
    
    try:
        # Run the main_orchestrator.py script.
        # The CWD for the subprocess will be PROJECT_ROOT.
        process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        process.wait()  # Wait for the orchestrator process to complete.
    except FileNotFoundError:
        print(f"Error: Failed to execute command. Ensure '{command[0]}' is a valid command or in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running MainOrchestrator: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication interrupted by user (Ctrl+C in run.py).")
    finally:
        print("Vtuber-AI application has finished.")

if __name__ == "__main__":
    main()
