# control_panel/app.py

import gradio as gr
import multiprocessing as mp
import threading
import time
import json
import os
from queue import Empty
from dotenv import load_dotenv
from src.utils import logger as app_logger
from src.utils.env_utils import setup_project_root


# --- Setup Project Path ---
# This ensures we can import from the 'src' directory
project_root = setup_project_root()
app_logger.setup_logging()
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)
# --- Import Your Existing Modules ---
# We need to import the orchestrator and the config loader
from src.process_orchestrator import ProcessOrchestrator

CONFIG_PATH = os.path.join(project_root, 'src/common/config.json')
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
        self.livechat_toggle_event = mp.Event()

        self.full_text = ""
        self.last_text_received_time = None # For detecting pauses between outputs

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
            livechat_toggle_event=self.livechat_toggle_event,
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
        
        if self.llm_output_display_queue.empty():
            return self.full_text

        now = time.time()
        if self.full_text and self.last_text_received_time and (now - self.last_text_received_time > 1.5):
            self.full_text += f"\n\n{'='*40}\n\n"

        try:
            while not self.llm_output_display_queue.empty():
                self.full_text += self.llm_output_display_queue.get_nowait()
        except Empty:
            pass

        self.last_text_received_time = now
        return self.full_text

    def clear_llm_output(self):
            """Clears the displayed LLM output text and resets the component state."""
            print("Clearing LLM output display.")
            self.full_text = ""
            self.last_text_received_time = None
            
            # Drain the queue to prevent old, unprocessed text from reappearing
            while not self.llm_output_display_queue.empty():
                try:
                    self.llm_output_display_queue.get_nowait()
                except Empty:
                    break
                    
            return "" # Return empty string to update the Gradio textbox


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
                    livechat_toggle_button = gr.Checkbox(label="💬 Enable Live Chat Fetching", value=False)
                
                terminate_job_button = gr.Button("🗑️ Terminate Current LLM/TTS Job")
                clear_output_button = gr.Button("🧹 Clear Output")

                llm_output_box = gr.Textbox(
                    label="LLM Live Output", 
                    interactive=False, 
                    lines=15,
                    autoscroll=True
                )

                # This will periodically update the output box
                timer = gr.Timer(0.2) 
                timer.tick(manager.poll_llm_output, None, llm_output_box, show_progress=False)

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
        clear_output_button.click(manager.clear_llm_output, None, llm_output_box)
        
        # Connect mute checkboxes
        stt_mute_button.change(lambda x: manager.stt_mute_event.set() if x else manager.stt_mute_event.clear(), inputs=[stt_mute_button], outputs=None)
        tts_mute_button.change(lambda x: manager.tts_mute_event.set() if x else manager.tts_mute_event.clear(), inputs=[tts_mute_button], outputs=None)
        livechat_toggle_button.change(lambda x: manager.livechat_toggle_event.set() if x else manager.livechat_toggle_event.clear(), inputs=[livechat_toggle_button], outputs=None)
    return demo


if __name__ == "__main__":
    # Ensure spawn start method is used for multiprocessing consistency
    mp.set_start_method('spawn', force=True)
    
    control_panel_ui = create_gradio_ui()
    control_panel_ui.launch()