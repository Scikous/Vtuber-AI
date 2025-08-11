# Install dependencies (if not done)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "trl<0.9.0" "bitsandbytes" "accelerate" "datasets"
# !pip install git+https://github.com/huggingface/transformers

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from unsloth import FastLanguageModel, FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(filename)s - %(funcName)s] %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================================
# 1️⃣ CONFIGURATION
# ======================================================================================
# Set your fine-tuning mode here: "language" or "vision"
FINETUNING_MODE = "language"

# You can also use command-line arguments for more flexibility
# parser = argparse.ArgumentParser(description="Unified Fine-tuning Script for Qwen2.5-VL")
# parser.add_argument("--mode", type=str, default="vision", choices=["language", "vision"], help="Set the fine-tuning mode.")
# args = parser.parse_args()
# FINETUNING_MODE = args.mode

# ======================================================================================
# 2️⃣ COMMON SETUP: Model ID, Dataset, and Paths
# ======================================================================================
MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
DATASET_PATH = "LLM_Wizard/dataset/final_templated_finetuning_data.parquet"
OUTPUT_DIR = f"LLM_Wizard/qwen2.5-vl-finetune-{FINETUNING_MODE}"
SAVE_DIR = f"LLM_Wizard/qwen2.5-vl-finetune-merged-{FINETUNING_MODE}"

# Load the dataset once
dataset = load_dataset("parquet", data_files={"train": DATASET_PATH})
logger.info(f"✅ Successfully loaded dataset from {DATASET_PATH}")

# ======================================================================================
# 3️⃣ MODE-SPECIFIC SETUP: Model Loading and PEFT Configuration
# ======================================================================================
if FINETUNING_MODE == "vision":
    logger.info("🚀 Initializing VISION fine-tuning mode...")
    # Load model and tokenizer using FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Set up PEFT for Vision + Language tuning
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_rslora=True,
        loftq_config=None,
        random_state=3407,
    )
    # Prepare model for vision training
    FastVisionModel.for_training(model)
    
    # Vision-specific trainer arguments
    trainer_specific_args = {
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
    }
    config_specific_args = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        "learning_rate": 2e-4,
        "fp16": not is_bf16_supported(),
        "bf16": is_bf16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "OUTPUT_DIR": OUTPUT_DIR,
        "report_to": "none",     # For Weights and Biases
        
        # You MUST put the below items for vision finetuning:
        "remove_unused_columns": False,
        "dataset_text_field": "",  # Must be empty for vision
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataset_num_proc": 4,
        "max_seq_length": 2048,
    }
    
elif FINETUNING_MODE == "language":
    logger.info("🚀 Initializing LANGUAGE fine-tuning mode...")
    # Load model and tokenizer using FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Set up PEFT for Language-only tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_rslora=True,
        loftq_config=None,
        random_state=3407,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Required formatting function for language mode
    def formatting_prompts_func(examples):
        return examples["text"]

    # Language-specific trainer arguments
    trainer_specific_args = {
        "formatting_func": formatting_prompts_func,
    }
    config_specific_args = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": 30,
        # num_train_epochs = 1, # Set this instead of max_steps for full training runs
        "learning_rate": 2e-4,
        "fp16": not is_bf16_supported(),
        "bf16": is_bf16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 3407,
        "OUTPUT_DIR": OUTPUT_DIR,
        "report_to": "none",     # For Weights and Biases
    # max_seq_length=2048,

    }
else:
    raise ValueError(f"Invalid FINETUNING_MODE: '{FINETUNING_MODE}'. Must be 'language' or 'vision'.")

# ======================================================================================
# 4️⃣ UNIFIED TRAINER SETUP
# ======================================================================================
# Common SFTConfig arguments
config_args = SFTConfig(
    **config_specific_args,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    **trainer_specific_args,
    args=config_args,
)

# ======================================================================================
# 5️⃣ TRAIN AND SAVE
# ======================================================================================
logger.info(f"🏁 Starting training for {FINETUNING_MODE} mode...")
trainer.train()
logger.info("✅ Training complete.")

# Save the final merged model
logger.info(f"💾 Saving merged model to {SAVE_DIR}...")
model.save_pretrained_merged(SAVE_DIR, tokenizer, save_method="merged_16bit")
logger.info(f"✅ Model successfully saved to {SAVE_DIR}")