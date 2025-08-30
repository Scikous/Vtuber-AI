# --- START OF FILE gaming.py ---
from models import VtuberExllamav2, LLMModelConfig
from huggingface_hub import snapshot_download
from model_utils import load_character, prompt_wrapper, contains_sentence_terminator
from time import perf_counter, strftime, gmtime
import asyncio
import logging
from collections import deque
import json
import os

import multiprocessing as mp
from queue import Empty
from PIL import Image, ImageDraw
from LLM_Wizard.gaming.game_capture import GameCaptureWorker
from LLM_Wizard.gaming.controls import InputController


# --- CONFIGURATION AND SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---!! DEBUGGING FLAG !! ---
# Set to True to save a visual of every mouse action to the 'debug_frames' folder.
DEBUG_VISUALIZE_ACTIONS = True
DEBUG_OUTPUT_DIR = "debug_frames"

# Character and Model Configuration
character_info_json = "LLM_Wizard/characters/character.json"
instructions, user_name, character_name = load_character(character_info_json)
main_model = "turboderp/Qwen2.5-VL-7B-Instruct-exl2"
tokenizer_model = "Qwen/Qwen2.5-VL-7B-Instruct"
revision = "8.0bpw"

# --- PROMPT TEMPLATES ---
PLANNER_PROMPT_TEMPLATE = """
You are "Strategos," a master video game strategist AI. Your role is to be the conscious, planning mind of an AI agent. You only process structured data.
Your task is to analyze the detailed `state_report` (provided by your Executor after it completed its last task) and your own `memory_log`. Based on this analysis, you will update your memory and determine the single most important `next_desire`.
Your response MUST be a single JSON object containing the `updated_memory_log` and the `next_desire`.

**Memory Log (JSON):**
{memory_log_json}

**State Report from Executor (JSON):**
{state_report_json}
"""

EXECUTOR_PROMPT_TEMPLATE = """
You are "Executor," a tactical AI agent with vision. Your purpose is to execute a high-level `desire`. You operate in one of three modes: EXECUTE, VERIFY, or REPORT.

**Current High-Level Desire:**
{current_desire}

**Task Mode:**
{task_mode}

---
**[IF TASK MODE IS 'EXECUTE']**
Your task is to determine the single, immediate next action to progress the desire. Analyze the `game_image` and provide your best estimate for the coordinates if a mouse action is needed (You must use target_x and target_y for the names of the coordinates, and their values must be integers). The output must be minimal for low latency.

**Format:**
{{
  "status": "<'in_progress', 'success', or 'failed'>",
  "action": {{
    "type": "<action_type>",
    "details": {{
      "button": "<e.g., 'left', 'W'>",
      "target_x": <integer_or_null>,
      "target_y": <integer_or_null>
    }}
  }}
}}
---
**[IF TASK MODE IS 'VERIFY']**
An action has been proposed. A red and black circle has been drawn on the `game_image` at the proposed `(x, y)` coordinate. Your task is to verify this action.
1.  Is the circle accurately placed on the intended target described by the `desire`?
2.  If it is NOT accurately placed, provide the corrected `(x, y)` coordinates.

**Format:**
{{
  "is_correct": <true_or_false>,
  "corrected_x": <integer_or_null>,
  "corrected_y": <integer_or_null>
}}

---
**[IF TASK MODE IS 'REPORT']**
Your task is complete. Analyze the final `game_image` and provide a detailed report for the Strategic Planner.

**Format:**
{{
  "task_completion_status": "<'success' or 'failed'>",
  "final_screen_classification": "<e.g., 'gameplay_hud'>",
  "summary_of_changes": "<A brief summary of what was accomplished.>",
  "extracted_text_and_elements": []
}}
---
"""

# --- HELPER FUNCTIONS ---

def parse_json_from_llm(llm_output: str, assistant_prompt: str, expected_keys: list) -> dict | None:
    """Safely parses a JSON string from the LLM output by prepending the assistant prompt."""
    full_json_str = assistant_prompt + llm_output
    try:
        data = json.loads(full_json_str)
        if not all(key in data for key in expected_keys):
            logging.warning(f"Parsed JSON is missing expected keys. Got: {data.keys()}")
            return None
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM output:\n---\n{full_json_str}\n---")
        return None

def draw_verification_marker(image: Image.Image, x: int, y: int) -> Image.Image:
    """Draws a visible circle marker on a copy of the image for verification."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    radius = 15
    p1 = (x - radius, y - radius)
    p2 = (x + radius, y + radius)
    draw.ellipse(p1 + p2, fill=None, outline="yellow", width=4)
    draw.ellipse(p1 + p2, fill=None, outline="green", width=2)
    img_copy.save(f"debug_frames/verify{p1}{p2}.png")
    return img_copy

async def execute_action(action: dict, controller: InputController):
    """
    Executes a game action by dispatching it to the InputController.
    Runs the synchronous controller method in a separate thread to avoid
    blocking the asyncio event loop.
    """
    if not action:
        logging.warning("execute_action received an empty action.")
        return
        
    # asyncio.to_thread is the modern way to run blocking I/O in an async app
    await asyncio.to_thread(controller.execute_action, action)

# --- MAIN AGENT LOGIC ---

async def run_gaming_agent():
    """Main async function to run the Planner/Executor gaming agent."""
    controller = InputController()
    images_dequeue = deque(maxlen=10)
    image_data_queue = mp.Queue()
    command_queue = mp.Queue()
    stop_event = mp.Event()

    capture_worker = GameCaptureWorker(
        image_data_queue=image_data_queue, command_queue=command_queue, stop_event=stop_event,
        interval_sec=0.2, source_type=1, target_size=(1024, 576)
    )

    try:
        capture_worker.start()
        logging.info(f"Started capture worker with PID: {capture_worker.pid}")

        model_config = LLMModelConfig(
            main_model=main_model, tokenizer_model=tokenizer_model, revision=revision,
            character_name=character_name, instructions=instructions, is_vision_model=True
        )

        agent_mode = "PLANNING"
        memory_log = {
            "game_title": "Unknown", "controls_mapped": False,
            "current_objective": "Determine the current situation and decide on a goal.",
            "notes": ["Agent started. Initializing first plan."]
        }
        current_desire = "No desire set yet."
        state_report_from_executor = None

        async with await VtuberExllamav2.load_model(config=model_config) as Character:
            logging.info("Model loaded. Starting main agent loop.")
            
            while not stop_event.is_set():
                # (Image queue processing is identical)
                if agent_mode == "PLANNING":
                    # (Planning phase is identical to the previous version)
                    logging.info("--- AGENT MODE: PLANNING ---")
                    prompt = PLANNER_PROMPT_TEMPLATE.format(
                        memory_log_json=json.dumps(memory_log, indent=2),
                        state_report_json=json.dumps(state_report_from_executor, indent=2)
                    )
                    assistant_prompt = '{\n  "updated_memory_log": {'
                    response_gen = await Character.dialogue_generator(
                        prompt=prompt, assistant_prompt=assistant_prompt, images=None,
                        max_tokens=1024, add_generation_prompt=False, continue_final_message=True
                    )
                    full_output = "".join([res.get("text", "") async for res in response_gen])
                    parsed_response = parse_json_from_llm(full_output, assistant_prompt, ["updated_memory_log", "next_desire"])
                    if parsed_response:
                        memory_log, current_desire = parsed_response["updated_memory_log"], parsed_response["next_desire"]
                        logging.info(f"New Desire from Planner: {current_desire}")
                        agent_mode = "EXECUTING"
                    else:
                        logging.error("Failed to parse Planner response. Retrying in 10 seconds."); await asyncio.sleep(10)
                        continue
                    
                # while not image_data_queue.empty():
                #     try:
                #         data_packet = image_data_queue.get_nowait()
                #         images_dequeue.append(Image.frombytes(data_packet['mode'], data_packet['size'], data_packet['image_bytes']))
                #     except Empty: break
                # if not images_dequeue: await asyncio.sleep(1); continue


                if agent_mode == "EXECUTING":
                    logging.info(f"--- AGENT MODE: EXECUTING (Desire: {current_desire}) ---")
                    task_finished = False
                    while not task_finished:
                        # (Image queue processing for latest frame is identical)
                        while not image_data_queue.empty():
                            data_packet = image_data_queue.get_nowait()
                            images_dequeue.append(Image.frombytes(data_packet['mode'], data_packet['size'], data_packet['image_bytes']))
                        if not images_dequeue: await asyncio.sleep(0.5); continue
                        latest_frame = images_dequeue[-1]

                        # --- Step 1: EXECUTE to get initial action ---
                        prompt = EXECUTOR_PROMPT_TEMPLATE.format(current_desire=current_desire, task_mode="EXECUTE")
                        assistant_prompt = '{\n  "status": "'
                        response_gen = await Character.dialogue_generator(
                            prompt=prompt, assistant_prompt=assistant_prompt, images=[latest_frame],
                            max_tokens=256, add_generation_prompt=False, continue_final_message=True
                        )
                        full_output = "".join([res.get("text", "") async for res in response_gen])
                        parsed_action = parse_json_from_llm(full_output, assistant_prompt, ["status", "action"])

                        if not parsed_action:
                            logging.error("Failed to parse initial action. Breaking task."); task_finished = True; continue
                        
                        action = parsed_action["action"]
                        action_type = action.get("type")
                        details = action.get("details", {})

                        # --- Step 2: VERIFY if it's a mouse action ---
                        # print("WHAT THE FUCK", action_type, details)
                        if action_type in ["mouse_move", "mouse_click", "click"] and "target_x" in details and "target_y" in details:
                            x,y = details["target_x"], details["target_y"]
                            try:
                                int(x)
                                int(y)
                            except ValueError:
                                continue
                            logging.info(f"Initial guess for {action_type}: ({details['target_x']}, {details['target_y']}). Verifying...")
                            
                            annotated_frame = draw_verification_marker(latest_frame, int(details['target_x']), int(details['target_y']))
                            
                            prompt = EXECUTOR_PROMPT_TEMPLATE.format(current_desire=current_desire, task_mode="VERIFY")
                            assistant_prompt = '{\n  "is_correct": '
                            response_gen = await Character.dialogue_generator(
                                prompt=prompt, assistant_prompt=assistant_prompt, images=[annotated_frame],
                                max_tokens=128, add_generation_prompt=False, continue_final_message=True
                            )
                            full_output = "".join([res.get("text", "") async for res in response_gen])
                            parsed_verification = parse_json_from_llm(full_output, assistant_prompt, ["is_correct"])
                            print("OH COME THE FUCK ON!!!!", parsed_verification)

                            if parsed_verification and not parsed_verification["is_correct"]:
                                new_x = parsed_verification.get("target_x")
                                new_y = parsed_verification.get("target_y")
                                if new_x is not None and new_y is not None:
                                    logging.info(f"Correction received. New coordinates: ({new_x}, {new_y})")
                                    action["details"]["target_x"] = new_x
                                    action["details"]["target_y"] = new_y
                                else:
                                    logging.warning("LLM indicated incorrect placement but failed to provide valid new coordinates.")
                            elif not parsed_verification:
                                logging.error("Failed to parse verification response. Proceeding with original coordinates.")

                        # --- Step 3: Execute final action ---
                        await execute_action(action, controller)
                        
                        if parsed_action["status"] in ["success", "failed"]:
                            logging.info(f"Executor task finished with status: {parsed_action['status']}")
                            task_finished = True
                        else:
                            await asyncio.sleep(0.5)

                    # (Reporting phase is identical to the previous version)
                    logging.info("--- EXECUTOR MODE: REPORTING ---")
                    final_frame = images_dequeue[-1]
                    prompt = EXECUTOR_PROMPT_TEMPLATE.format(current_desire=current_desire, task_mode="REPORT")
                    assistant_prompt = '{\n  "task_completion_status": "'
                    response_gen = await Character.dialogue_generator(
                        prompt=prompt, assistant_prompt=assistant_prompt, images=[final_frame],
                        max_tokens=1024, add_generation_prompt=False, continue_final_message=True
                    )
                    full_output = "".join([res.get("text", "") async for res in response_gen])
                    state_report_from_executor = parse_json_from_llm(full_output, assistant_prompt, ["task_completion_status", "summary_of_changes"])
                    if not state_report_from_executor:
                        state_report_from_executor = {
                            "task_completion_status": "failed",
                            "summary_of_changes": "Executor failed to generate a valid report after completing its task.",
                            "extracted_text_and_elements": []
                        }
                    agent_mode = "PLANNING"

    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, signaling worker to stop.")
    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}", exc_info=True)
    finally:
        # (Cleanup is identical)
        logging.info("Cleaning up...")
        if 'capture_worker' in locals() and capture_worker.is_alive():
            stop_event.set()
            # Close the controller first
            if controller:
                controller.close()
            capture_worker.join(timeout=5)
            if capture_worker.is_alive():
                logging.warning("Worker did not terminate gracefully, terminating.")
                capture_worker.terminate()
        logging.info("Cleanup complete.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(run_gaming_agent())

