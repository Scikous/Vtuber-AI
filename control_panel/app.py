# # control_panel/app.py

# import gradio as gr
# import sys
# import os

# # Add the project root to the Python path to allow importing from 'src'
# # This is a common practice when running scripts in subdirectories
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # Now you can import your workers
# # Assuming you have functions like process_stt, process_llm, process_tts in your workers
# # from src.worker.stt_worker import process_stt
# # from src.worker.llm_worker import process_llm
# # from src.worker.tts_worker import process_tts

# # --- Mock functions for demonstration ---
# # Replace these with your actual worker functions
# def mock_process_stt(audio_file_path):
#     """Mock STT worker that returns transcribed text."""
#     print(f"STT processing: {audio_file_path}")
#     return "This is a test transcription."

# def mock_process_llm(text_input):
#     """Mock LLM worker that returns a generated response."""
#     print(f"LLM processing: {text_input}")
#     return f"The LLM received: '{text_input}' and is generating a response."

# def mock_process_tts(text_input):
#     """Mock TTS worker that returns a path to an audio file."""
#     print(f"TTS processing: {text_input}")
#     # In a real scenario, this would generate an audio file and return its path
#     # For this example, we'll return a placeholder path
#     return "path/to/generated_speech.wav"
# # -----------------------------------------

# def stt_llm_tts_pipeline(audio_input):
#     """
#     The main function to connect the pipeline components for the Gradio interface.
#     """
#     transcribed_text = mock_process_stt(audio_input)
#     llm_response = mock_process_llm(transcribed_text)
#     output_audio_path = mock_process_tts(llm_response)
    
#     return transcribed_text, llm_response, output_audio_path

# # --- Create the Gradio Interface ---
# with gr.Blocks() as demo:
#     gr.Markdown("# STT -> LLM -> TTS Control Panel")
    
#     with gr.Row():
#         with gr.Column():
#             audio_input = gr.Audio(type="filepath", label="Speak Here")
#             submit_button = gr.Button("Process Audio")
        
#         with gr.Column():
#             transcribed_output = gr.Textbox(label="Transcribed Text (STT)")
#             llm_output = gr.Textbox(label="LLM Response")
#             tts_output = gr.Audio(label="Spoken Response (TTS)")
            
#     submit_button.click(
#         fn=stt_llm_tts_pipeline,
#         inputs=audio_input,
#         outputs=[transcribed_output, llm_output, tts_output]
#     )

#     gr.Markdown("## Examples")
#     gr.Examples(
#         examples=[
#             # Provide paths to example audio files if you have them
#             # "path/to/example1.wav",
#         ],
#         inputs=audio_input,
#         outputs=[transcribed_output, llm_output, tts_output],
#         fn=stt_llm_tts_pipeline,
#         cache_examples=True
#     )


# # --- Launch the App ---
# if __name__ == "__main__":
#     # The launch() method starts a simple web server. [4]
#     # server_name="0.0.0.0" makes it accessible on your local network
#     demo.launch(server_name="0.0.0.0", server_port=8888) 


# control_panel/app.py

import gradio as gr
import multiprocessing as mp
import threading
import time
import json
import os
import sys
from queue import Empty

# --- Setup Project Path ---
# This ensures we can import from the 'src' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Your Existing Modules ---
# We need to import the orchestrator and the config loader
from src.process_orchestrator import ProcessOrchestrator
from src.common import config as app_config

CONFIG_PATH = os.path.join(project_root, 'src/common/config.json')
print("WHWHWHWHW", CONFIG_PATH)
class AppManager:
    """A singleton class to manage the lifecycle of the ProcessOrchestrator and its workers."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        
        self.orchestrator: ProcessOrchestrator = None
        self.orchestrator_thread: threading.Thread = None

        # --- Inter-Process Communication Objects ---
        # These will be shared between the Gradio app and the worker processes
        self.llm_output_display_queue = mp.Queue()
        self.tts_mute_event = mp.Event()
        self.stt_mute_event = mp.Event()
        
        self.initialized = True
        print("AppManager Initialized.")

    def _load_config(self):
        """Loads configuration from the JSON file."""
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)

    def _save_config(self, config_data):
        """Saves configuration to the JSON file."""
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)

    def is_running(self):
        return self.orchestrator_thread is not None and self.orchestrator_thread.is_alive()

    def start_workers(self):
        if self.is_running():
            print("Workers are already running.")
            return

        print("Starting worker processes...")
        # We create a new ProcessOrchestrator instance, passing it our control objects
        self.orchestrator = ProcessOrchestrator(
            llm_output_display_queue=self.llm_output_display_queue,
            tts_mute_event=self.tts_mute_event,
            stt_mute_event=self.stt_mute_event,
            is_managed=True
        )
        self.orchestrator_thread = threading.Thread(target=self.orchestrator.run, daemon=True)
        self.orchestrator_thread.start()
        print("Orchestrator thread started.")
        return "Workers started."

    def stop_workers(self):
        if not self.is_running():
            print("Workers are not running.")
            return "Workers already stopped."
        
        print("Stopping worker processes...")
        self.orchestrator.shutdown_event.set()
        self.orchestrator_thread.join(timeout=20)
        
        if self.orchestrator_thread.is_alive():
            print("Warning: Orchestrator thread did not terminate gracefully.")
        
        self.orchestrator = None
        self.orchestrator_thread = None
        print("Workers stopped.")
        return "Workers stopped."

    def restart_workers(self, config_data):
        print("Restarting workers...")
        self.stop_workers()
        # Give processes time to release resources
        time.sleep(2)
        # Save the new config before starting again
        self._save_config(config_data)
        self.start_workers()
        print("Workers restarted with new configuration.")
        return "Workers restarted."

    def terminate_current_job(self):
        if not self.is_running():
            return "Workers are not running."
        
        terminate_msg = self._load_config().get("TERMINATE_OUTPUT", "w1zt3r")
        # The llm_to_tts_queue is owned by the orchestrator, so we access it there
        self.orchestrator.queues["llm_to_tts_queue"].put(terminate_msg)
        return f"Termination signal '{terminate_msg}' sent."

    def poll_llm_output(self):
        """Polls the queue for new text from the LLM to display in the UI."""
        if not self.is_running():
            return ""
        
        full_text = ""
        try:
            while not self.llm_output_display_queue.empty():
                full_text += self.llm_output_display_queue.get_nowait()
        except Empty:
            pass
        return full_text

# --- Instantiate the Manager ---
manager = AppManager()

# --- Gradio UI Definition ---
def create_gradio_ui():
    initial_config = manager._load_config()

    with gr.Blocks(theme=gr.themes.Soft(), title="Pipeline Control Panel") as demo:
        gr.Markdown("# Pipeline Control Panel")

        with gr.Tabs():
            # --- Main Control Tab ---
            with gr.TabItem("Main Controls & Output"):
                with gr.Row():
                    start_button = gr.Button("🚀 Start System", variant="primary")
                    stop_button = gr.Button("🛑 Shutdown System", variant="stop")
                    
                with gr.Row():
                    stt_mute_button = gr.Checkbox(label="🎤 Mute Microphone (STT)", value=False)
                    tts_mute_button = gr.Checkbox(label="🔊 Mute Speaker (TTS)", value=False)
                
                terminate_job_button = gr.Button("🗑️ Terminate Current LLM/TTS Job")
                
                llm_output_box = gr.Textbox(
                    label="LLM Live Output", 
                    interactive=False, 
                    lines=15,
                    autoscroll=True
                )
                # This will periodically update the output box
                demo.load(manager.poll_llm_output, None, llm_output_box, show_progress=False)

            # --- Settings Tabs ---
            with gr.TabItem("⚙️ System Settings"):
                gr.Markdown("## Worker Configuration")
                gr.Markdown("Click 'Apply & Restart' to save changes. The system will restart to load new models and settings.")
                
                # We use gr.State to hold the config dict
                config_state = gr.State(initial_config)

                with gr.Accordion("LLM Settings", open=False):
                    llm_model_path = gr.Textbox(label="Main LLM Model Path", value=initial_config["llm_settings"]["llm_model_path"])
                    llm_model_revision = gr.Textbox(label="Main LLM Model Revision", value=initial_config["llm_settings"]["llm_model_revision"])
                    llm_max_tokens = gr.Slider(256, 4096, step=128, label="Max Tokens", value=initial_config["llm_settings"]["max_tokens"])

                with gr.Accordion("Context LLM Settings", open=False):
                    context_llm_model_path = gr.Textbox(label="Context LLM Model Path", value=initial_config["context_llm_settings"]["context_llm_model_path"])
                    context_llm_model_revision = gr.Textbox(label="Context LLM Model Revision", value=initial_config["context_llm_settings"]["context_llm_model_revision"])
                
                with gr.Accordion("TTS Settings", open=False):
                    tts_engine = gr.Dropdown(label="TTS Engine", choices=["coqui", "piper"], value=initial_config["tts_settings"]["tts_engine"])
                    voice_to_clone_file = gr.File(label="Voice to Clone (Optional .wav file)", type="filepath")

                with gr.Accordion("STT Settings", open=False):
                    stt_model_size = gr.Dropdown(label="STT Model Size", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"], value=initial_config["stt_settings"]["MODEL_SIZE"])
                    stt_language = gr.Textbox(label="Language", value=initial_config["stt_settings"]["LANGUAGE"])
                    stt_silence_duration = gr.Slider(0.1, 2.0, step=0.1, label="Silence Duration to Finalize (s)", value=initial_config["stt_settings"]["SILENCE_DURATION_S_FOR_FINALIZE"])

                apply_button = gr.Button("💾 Apply & Restart Workers", variant="primary")

        # --- UI Logic Connections ---
        
        # Function to update the config dict from UI elements
        def update_config_from_ui(
            llm_path, llm_rev, llm_tokens, 
            ctx_path, ctx_rev, 
            tts_eng, tts_voice,
            stt_model, stt_lang, stt_silence,
            current_config):
            
            current_config["llm_settings"]["llm_model_path"] = llm_path
            current_config["llm_settings"]["llm_model_revision"] = llm_rev
            current_config["llm_settings"]["max_tokens"] = llm_tokens
            current_config["context_llm_settings"]["context_llm_model_path"] = ctx_path
            current_config["context_llm_settings"]["context_llm_model_revision"] = ctx_rev
            current_config["tts_settings"]["tts_engine"] = tts_eng
            current_config["tts_settings"]["voice_to_clone_file"] = tts_voice # Add new field
            current_config["stt_settings"]["MODEL_SIZE"] = stt_model
            current_config["stt_settings"]["LANGUAGE"] = stt_lang
            current_config["stt_settings"]["SILENCE_DURATION_S_FOR_FINALIZE"] = stt_silence
            
            manager.restart_workers(current_config)
            
            # Show a confirmation and return the updated config to the state
            gr.Info("Configuration saved. Workers are restarting...")
            return current_config

        ui_components = [
            llm_model_path, llm_model_revision, llm_max_tokens,
            context_llm_model_path, context_llm_model_revision,
            tts_engine, voice_to_clone_file,
            stt_model_size, stt_language, stt_silence_duration
        ]

        apply_button.click(
            fn=update_config_from_ui,
            inputs=ui_components + [config_state],
            outputs=[config_state]
        )

        start_button.click(manager.start_workers, None, None)
        stop_button.click(manager.stop_workers, None, None)
        terminate_job_button.click(manager.terminate_current_job, None, None)
        
        # Connect mute checkboxes
        stt_mute_button.change(lambda x: manager.stt_mute_event.set() if x else manager.stt_mute_event.clear(), inputs=[stt_mute_button], outputs=None)
        tts_mute_button.change(lambda x: manager.tts_mute_event.set() if x else manager.tts_mute_event.clear(), inputs=[tts_mute_button], outputs=None)

    return demo


if __name__ == "__main__":
    # Ensure spawn start method is used for multiprocessing consistency
    mp.set_start_method('spawn', force=True)
    
    control_panel_ui = create_gradio_ui()
    control_panel_ui.launch()