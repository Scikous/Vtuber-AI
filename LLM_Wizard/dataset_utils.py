import sys
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
import json # For model_utils placeholder
from model_utils import load_character, prompt_wrapper
import os

# --- Configuration & Setup ---
# MODEL_PATH = "NousResearch/Meta-Llama-3-8B" # Using a more common model for example
# If you have a local model:
# MODEL_PATH = "unsloth/Meta-Llama-3.1-8B"
# MODEL_PATH = 'LLM/Meta-Llama-3.1-8B/'

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"#"LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"
try:
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
except:
    # This is a placeholder for environments where snapshot_download might be used
    # from huggingface_hub import snapshot_download
    # hf_model_path = snapshot_download(repo_id=MODEL_PATH)
    # TOKENIZER = AutoTokenizer.from_pretrained(hf_model_path)
    print(f"Could not load tokenizer for {MODEL_PATH}. Ensure you are logged in or the model is available.")
    sys.exit(1)


# Load character details (globally or pass as needed)
CHARACTER_JSON_PATH = 'LLM_Wizard/characters/character.json'
# Dummy character loading will be handled in main() if file doesn't exist
INSTRUCTIONS, USER_NAME, CHARACTER_NAME = "", "", ""
# --- NEW: Configuration for max conversation turns ---
MAX_TURNS = 12 # Set the limit. Use None for no limit.


# --- Core Functions ---
def prepare_finetuning_input_parquet(csv_path: str, output_parquet_path: str):
    """
    Reads a CSV file, extracts user, character, context, and conversation_id columns,
    and saves them to a Parquet file for later chat templating.
    """
    try:
        df = pd.read_csv(csv_path)
        df = df.fillna('') # Handle potential missing values

        # Ensure required columns exist
        required_cols = ['user', 'character', 'context', 'conversation_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV file {csv_path} is missing required columns: {missing_cols}")

        df_finetuning = df[required_cols]
        df_finetuning.to_parquet(output_parquet_path, engine="pyarrow")
        print(f"Successfully saved raw fine-tuning data to {output_parquet_path}")
        print(f"Columns in raw fine-tuning data: {df_finetuning.columns.tolist()}")
        if not df_finetuning.empty:
            print("First row of raw fine-tuning data:")
            print(df_finetuning.head(1))
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in prepare_finetuning_input_parquet: {e}")

def _build_conversation_messages(conversation_data: list, instructions: str):
    """
    Helper function to build messages array from conversation data.
    conversation_data: List of dictionaries with 'user', 'character', 'context' keys, ordered by conversation flow.
    """
    messages = [{"role": "system", "content": instructions}]
    
    for turn in conversation_data:
        prompt = prompt_wrapper(turn["user"], turn["context"])
        messages.append({"role": "user", "content": prompt.strip()})
        messages.append({"role": "assistant", "content": turn["character"]})
    
    return messages


# def _build_conversation_messages(conversation_data: list, instructions: str):
# """
# Helper function to build messages array from conversation data.
# conversation_data: List of dictionaries with 'user', 'character', 'context' keys, ordered by conversation flow.
# """
# messages = [{"role": "system", "content": instructions}]

# for turn in conversation_data:
#     prompt = prompt_wrapper(turn["user"], turn["context"])
#     messages.append({"role": "user", "content": [{"type": "text", "text" : prompt.strip()},
#                                                  {"type": "image", "image" : "nan"}]})
#     messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["character"]}]})

# return messages

def prepare_exllamav2_calibration_parquet(csv_path: str, output_parquet_path: str):
    """
    Reads a CSV file, extracts the 'user' column, renames it to 'text',
    and saves it to a Parquet file for ExLlamaV2 calibration.
    """
    try:
        df = pd.read_csv(csv_path)
        df = df.fillna('') # Handle potential missing values

        if 'user' not in df.columns:
            raise ValueError(f"CSV file {csv_path} is missing the required 'user' column.")

        df_calibration = df[['user']].rename(columns={'user': 'text'})
        df_calibration.to_parquet(output_parquet_path, engine="pyarrow")
        print(f"Successfully saved ExLlamaV2 calibration data to {output_parquet_path}")
        if not df_calibration.empty:
            print("First few rows of calibration data ('text' column):")
            print(df_calibration.head())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred in prepare_exllamav2_calibration_parquet: {e}")


def _apply_chat_template_func(example: dict, instructions: str, tokenizer: AutoTokenizer):
    """
    Helper function to apply chat template to a conversation example.
    'example' is expected to have a 'conversation' key containing a list of turns.
    """
    messages = _build_conversation_messages(example["conversation"], instructions)
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt" #off for tokenize False
    )
    return {"text": text}

def _group_conversations_by_id(dataset, max_turns: int = None):
    """
    Groups dataset rows by conversation_id. If a conversation exceeds `max_turns`,
    it generates multiple overlapping "sliding window" examples.
    Returns a new dataset where each row is a valid training conversation.
    """
    df = dataset.to_pandas()
    all_generated_conversations = []

    for conv_id, group in df.groupby('conversation_id'):
        # --- FIX: Ensure all conversation IDs are strings to prevent type errors ---
        conv_id = str(conv_id)
        # --- END FIX ---

        group = group.sort_index()
        num_turns = len(group)

        # If the conversation is longer than max_turns, create sliding windows
        if max_turns and num_turns > max_turns:
            num_windows = num_turns - max_turns + 1
            print(f"Conversation '{conv_id}' has {num_turns} turns. Generating {num_windows} sliding window examples of max size {max_turns}.")
            
            for i in range(num_windows):
                window_df = group.iloc[i : i + max_turns]
                
                conversation_turns = []
                for _, row in window_df.iterrows():
                    conversation_turns.append({
                        'user': row['user'],
                        'character': row['character'],
                        'context': row['context']
                    })
                
                # Create a new, unique ID for each generated window
                new_conv_id = f"{conv_id}_window_{i+1}"
                all_generated_conversations.append({
                    'conversation_id': new_conv_id,
                    'conversation': conversation_turns
                })
        else:
            # If conversation is within the limit, process it as a single block
            conversation_turns = []
            for _, row in group.iterrows():
                conversation_turns.append({
                    'user': row['user'],
                    'character': row['character'],
                    'context': row['context']
                })
            
            # Now `conv_id` is guaranteed to be a string here as well
            all_generated_conversations.append({
                'conversation_id': conv_id,
                'conversation': conversation_turns
            })

    return Dataset.from_list(all_generated_conversations)



def load_and_apply_chat_template(
    raw_finetuning_parquet_path: str,
    instructions: str,
    tokenizer: AutoTokenizer,
    split: str = "train",
    max_turns: int = None
) -> Dataset:
    """
    Loads a Parquet file containing conversation data, groups by conversation_id,
    applies the chat template, and returns the processed Hugging Face Dataset.

    Args:
        raw_finetuning_parquet_path (str): Path to the input Parquet file.
        instructions (str): The system prompt for the character.
        tokenizer (AutoTokenizer): The tokenizer for applying the chat template.
        split (str): The dataset split to process (e.g., "train").
        max_turns (int, optional): The maximum number of turns to keep in a conversation.
                                   If a conversation exceeds this, the oldest turns are dropped.
                                   Defaults to None (no limit).
    """
    try:
        # Load the dataset
        dataset_dict = load_dataset("parquet", data_files={split: raw_finetuning_parquet_path})
        
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found in the loaded dataset. Available splits: {list(dataset_dict.keys())}")
        
        raw_dataset = dataset_dict[split]

        # Verify columns before processing
        required_cols = ['user', 'character', 'context', 'conversation_id']
        if not all(col in raw_dataset.column_names for col in required_cols):
            raise ValueError(
                f"Parquet file {raw_finetuning_parquet_path} is missing one or more required columns "
                f"for chat templating: {required_cols}. Found: {raw_dataset.column_names}"
            )

        # Group conversations by ID, applying the turn limit
        conversation_dataset = _group_conversations_by_id(raw_dataset, max_turns=max_turns)
        
        # Apply chat template to each conversation
        formatted_dataset = conversation_dataset.map(
            _apply_chat_template_func,
            fn_kwargs={'instructions': instructions, 'tokenizer': tokenizer}
        )
        
        print(f"Successfully applied chat template to {len(formatted_dataset)} conversations from {raw_finetuning_parquet_path}")
        return formatted_dataset
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {raw_finetuning_parquet_path}")
        return None
    except ValueError as ve:
        print(f"Error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in load_and_apply_chat_template: {e}")
        return None

# --- Main Execution ---
def main():
    # Define paths (consider making these configurable, e.g., via argparse)
    base_path = "LLM_Wizard/dataset/"
    csv_path = base_path + "John_Smith_Base.csv" # Your input CSV
    
    # --- END NEW ---
    
    # Create dummy CSV and character.json if they don't exist for the script to run
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(CHARACTER_JSON_PATH), exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Creating dummy CSV at {csv_path}")
        # Add more data to test the trimming logic
        dummy_data = {
            'user': [f"Turn {i}" for i in range(1, 18)] + ["Hi", "Hello again"],
            'character': [f"Response {i}" for i in range(1, 18)] + ["I am here.", "Yes?"],
            'context': [f"Context for turn {i}" for i in range(1, 18)] + ["Greeting", "Follow-up"],
            'conversation_id': ["long_conv"] * 17 + ["short_conv"] * 2
        }
        pd.DataFrame(dummy_data).to_csv(csv_path, index=False)

    if not os.path.exists(CHARACTER_JSON_PATH):
        print(f"Creating dummy character JSON at {CHARACTER_JSON_PATH}")
        dummy_char_info = {
            "instructions": "You are Capybara, a witty and helpful AI assistant specialized in South American wildlife. You are speaking to {user_name}.",
            "user_name": "Researcher",
            "character_name": "Capybara"
        }
        with open(CHARACTER_JSON_PATH, 'w') as f:
            json.dump(dummy_char_info, f, indent=2)
    
    # Reload instructions now that we're sure the file exists
    global INSTRUCTIONS, USER_NAME, CHARACTER_NAME
    INSTRUCTIONS, USER_NAME, CHARACTER_NAME = load_character(CHARACTER_JSON_PATH)


    raw_finetuning_parquet_path = base_path + "finetuning_input.parquet"
    calibration_parquet_path = base_path + "exllama_calibration.parquet"
    final_templated_parquet_path = base_path + "final_templated_finetuning_data.parquet"


    print(f"Using tokenizer: {MODEL_PATH}")
    print(f"BOS token: '{TOKENIZER.bos_token}', EOS token: '{TOKENIZER.eos_token}'")
    print(f"System Instructions (loaded from {CHARACTER_JSON_PATH}):\n{INSTRUCTIONS}\n")

    # 1. Prepare the raw input Parquet for fine-tuning (user, character, context columns)
    prepare_finetuning_input_parquet(csv_path, raw_finetuning_parquet_path)

    # 2. Prepare the Parquet for ExLlamaV2 calibration (text column)
    prepare_exllamav2_calibration_parquet(csv_path, calibration_parquet_path)

    # 3. Load the raw fine-tuning data and apply the chat template
    print(f"\nApplying chat template with a max turn limit of: {MAX_TURNS}")
    templated_dataset = load_and_apply_chat_template(
        raw_finetuning_parquet_path,
        INSTRUCTIONS,
        TOKENIZER,
        max_turns=MAX_TURNS
    )

    if templated_dataset:
        print(f"\n--- Example of templated data (from {raw_finetuning_parquet_path}) ---")
        if len(templated_dataset) > 0:
            # Find and print the long conversation to verify trimming
            for example in templated_dataset:
                if example['conversation_id'] == 'long_conv':
                    print("\n--- Trimmed 'long_conv' example ---")
                    print(example['text'])
                    break
            else: # If loop finishes without break, print the first one
                print("\n--- First available example ---")
                print(templated_dataset[28]['text'])

            try:
                templated_dataset.to_parquet(final_templated_parquet_path)
                print(f"\nSuccessfully saved final templated dataset to {final_templated_parquet_path}")
            except Exception as e:
                print(f"Error saving final templated dataset: {e}")
        else:
            print("Templated dataset is empty.")
    else:
        print("Failed to generate templated dataset.")

if __name__ == "__main__":
    # Added a simple check for placeholder functions to allow standalone execution
    if 'model_utils' not in sys.modules:
        print("Creating dummy model_utils functions to run script.")
        def load_character(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return data.get("instructions", ""), data.get("user_name", ""), data.get("character_name", "")
        def prompt_wrapper(user_prompt, context):
            return f"{user_prompt}\n[Context: {context}]"
        
        # Inject them into the global scope so the script can find them
        sys.modules['model_utils'] = type('module', (object,), {
            'load_character': load_character,
            'prompt_wrapper': prompt_wrapper
        })()
        from model_utils import load_character, prompt_wrapper

    main()