import sys
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
import json # For LLMUtils placeholder
from model_utils import LLMUtils # character_loader
# --- Configuration & Setup ---
# MODEL_PATH = "NousResearch/Meta-Llama-3-8B" # Using a more common model for example
# If you have a local model:
MODEL_PATH = "LLM_Wizard/CapybaraHermes-2.5-Mistral-7B-GPTQ"
# MODEL_PATH = "unsloth/Meta-Llama-3.1-8B"
# MODEL_PATH = 'LLM/Meta-Llama-3.1-8B/'

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load character details (globally or pass as needed)
# Ensure 'LLM_Wizard/characters/character.json' exists or adjust path
# Create a dummy character.json if it doesn't exist for the script to run
# Example dummy character.json:
# {
#   "instructions": "You are Capybara, a friendly and knowledgeable assistant. You are talking to {user_name}.",
#   "user_name": "Explorer",
#   "character_name": "Capybara"
# }
CHARACTER_JSON_PATH = 'LLM_Wizard/characters/character.json'
INSTRUCTIONS, USER_NAME, CHARACTER_NAME = LLMUtils.load_character(CHARACTER_JSON_PATH)


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
        prompt = LLMUtils.prompt_wrapper(turn["user"], turn["context"])
        messages.append({"role": "user", "content": prompt.strip()})
        messages.append({"role": "assistant", "content": turn["character"]})
    
    return messages

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

        # It's good practice to search for this information if unsure.
        # Searching "ExLlamaV2 calibration dataset format" confirms it typically
        # expects a Parquet file with a single column named "text".
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
    
    formatted_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"formatted_chat": formatted_chat}


def _group_conversations_by_id(dataset):
    """
    Groups dataset rows by conversation_id and creates conversation objects.
    Returns a new dataset where each row represents a complete conversation.
    """
    # Convert to pandas for easier grouping
    df = dataset.to_pandas()
    
    conversations = []
    for conv_id, group in df.groupby('conversation_id'):
        # Sort by index to maintain conversation order (assuming data is pre-ordered)
        group = group.sort_index()
        
        conversation_turns = []
        for _, row in group.iterrows():
            conversation_turns.append({
                'user': row['user'],
                'character': row['character'],
                'context': row['context']
            })
        
        conversations.append({
            'conversation_id': conv_id,
            'conversation': conversation_turns
        })
    
    # Convert back to Dataset
    return Dataset.from_list(conversations)

def load_and_apply_chat_template(
    raw_finetuning_parquet_path: str,
    instructions: str,
    tokenizer: AutoTokenizer,
    split: str = "train"
) -> Dataset:
    """
    Loads a Parquet file containing conversation data, groups by conversation_id,
    applies the chat template, and returns the processed Hugging Face Dataset.
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

        # Group conversations by ID
        conversation_dataset = _group_conversations_by_id(raw_dataset)
        
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
    csv_path = base_path + "test.csv" # Your input CSV
    
    # Create dummy CSV and character.json if they don't exist for the script to run
    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(CHARACTER_JSON_PATH), exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Creating dummy CSV at {csv_path}")
        dummy_data = {
            'user': [
                "Hello, who are you?", 
                "That's interesting, tell me more about yourself.",
                "What is the capital of France?", 
                "Tell me a joke about capybaras.",
                "That was funny! Do you know any facts about capybaras?"
            ],
            'character': [
                "I am Capybara, your assistant.", 
                "I'm an AI designed to help with various tasks and provide information about wildlife, especially South American animals.",
                "The capital of France is Paris.", 
                "Why did the capybara cross the road? To get to the other tide!",
                "Yes! Capybaras are the world's largest rodents and are excellent swimmers. They're native to South America."
            ],
            'context': [
                "General introduction question.", 
                "Follow-up about assistant capabilities.",
                "This is a geography question.", 
                "This is a humor request.",
                "Follow-up question about capybara facts."
            ],
            'conversation_id': [
                "conv_1", 
                "conv_1",
                "conv_2", 
                "conv_3",
                "conv_3"
            ]
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
        # Reload instructions if dummy was created
        global INSTRUCTIONS, USER_NAME, CHARACTER_NAME
        INSTRUCTIONS, USER_NAME, CHARACTER_NAME = LLMUtils.load_character(CHARACTER_JSON_PATH)


    raw_finetuning_parquet_path = base_path + "finetuning_input.parquet"
    calibration_parquet_path = base_path + "exllama_calibration.parquet"
    # This will be the final dataset with chat templates applied, ready for training.
    # We can save it to disk if needed, or use it directly.
    final_templated_parquet_path = base_path + "final_templated_finetuning_data.parquet"


    print(f"Using tokenizer: {MODEL_PATH}")
    print(f"BOS token: '{TOKENIZER.bos_token}', EOS token: '{TOKENIZER.eos_token}'")
    print(f"System Instructions (loaded from {CHARACTER_JSON_PATH}):\n{INSTRUCTIONS}\n")

    # 1. Prepare the raw input Parquet for fine-tuning (user, character, context columns)
    prepare_finetuning_input_parquet(csv_path, raw_finetuning_parquet_path)

    # 2. Prepare the Parquet for ExLlamaV2 calibration (text column)
    prepare_exllamav2_calibration_parquet(csv_path, calibration_parquet_path)

    # 3. Load the raw fine-tuning data and apply the chat template
    # This function can be called from your fine-tuning script
    templated_dataset = load_and_apply_chat_template(
        raw_finetuning_parquet_path,
        INSTRUCTIONS,
        TOKENIZER
    )

    if templated_dataset:
        print(f"\n--- Example of templated data (from {raw_finetuning_parquet_path}) ---")
        if len(templated_dataset) > 0:
            print(templated_dataset[0]['formatted_chat'])
            # Optionally, save the fully processed dataset
            # For Hugging Face datasets, you can save in various formats, including Parquet
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
    main()