# import random
# import torch
# import numpy as np
# from datasets import DatasetDict, load_dataset
# from unsloth import FastLanguageModel # Added Unsloth import
# from transformers import AutoTokenizer # Removed AutoModelForCausalLM, BitsAndBytesConfig
# from peft import (
#     LoraConfig,
#     PeftModel,
#     TaskType,
#     # get_peft_model, # Removed, Unsloth handles this internally
#     # prepare_model_for_kbit_training, # Removed, Unsloth handles this internally
# )
# from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# SEED = 42
# PAD_TOKEN = "<|pad|>" # Unsloth might handle padding differently, check documentation if issues arise
# BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit" # Using Unsloth's optimized model
# NEW_MODEL = "LLM_Wizard/Qwen2.5-VL-7B-Unsloth"
# OUTPUT_DIR = "LLM_Wizard/Qwen2.5-VL-7B-Unsloth"
# DATASET_PATH = "LLM_Wizard/dataset/final_templated_finetuning_data.parquet"

# class ModelTrainer:
#     def __init__(self, base_model: str, new_model: str, pad_token: str, output_dir: str, dataset_path: str):
#         self.base_model = base_model
#         self.new_model = new_model
#         self.pad_token = pad_token # Keep for tokenizer setup, but Unsloth might override
#         self.output_dir = output_dir
#         self.dataset_path = dataset_path
#         self.tokenizer = None
#         self.model = None
#         self.dataset = None

#     @classmethod
#     def prepare_for_training(cls, base_model: str, new_model: str, pad_token: str, output_dir: str, dataset_path: str):
#         trainer = cls(base_model, new_model, pad_token, output_dir, dataset_path)
#         trainer.seed_everything(SEED)
#         trainer.dataset = trainer.load_and_split_dataset(dataset_path)
#         # Tokenizer and model loading are now combined in load_model_and_tokenizer
#         trainer.model, trainer.tokenizer = trainer.load_model_and_tokenizer()
#         # Lora preparation is handled by FastLanguageModel internally when adding adapters
#         # trainer.prepare_lora_model() # Removed
#         return trainer

#     def clear_memory(self):
#         import gc
#         gc.collect()
#         torch.cuda.empty_cache()

#     @staticmethod
#     def seed_everything(seed: int):
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)

#     # Combined model and tokenizer loading for Unsloth
#     def load_model_and_tokenizer(self):
#         print(f"Loading Unsloth model: {self.base_model}")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = self.base_model,
#             max_seq_length = 512, # Set max sequence length here
#             dtype = None, # None will default to torch.float16
#             load_in_4bit = True,
#         )
#         # Setup padding token if necessary, Unsloth might handle this
#         if tokenizer.pad_token is None:
#              tokenizer.add_special_tokens({"pad_token": self.pad_token})
#         tokenizer.padding_side = "right"
#         # No need to resize embeddings manually, Unsloth handles it
#         # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
#         return model, tokenizer

#     # Removed setup_tokenizer and load_model as they are combined now
#     # def setup_tokenizer(self):
#     #     ...
#     # def load_model(self):
#     #     ...

#     def load_and_split_dataset(self, data_file: str):
#         '''
#         Assumes that the datas has been formatted correctly -- usually done through dataset_creator.py

#         Also WIP, reads from a text file currently, should be parquet
#         '''
#         dataset = load_dataset("text", data_files=data_file)
#         test_size = 0.1
#         # Ensure consistent splitting
#         dataset = dataset['train'].train_test_split(test_size=test_size, seed=SEED)
#         train_val_test = dataset
#         # Further split the test set into validation and test
#         val_test_split = train_val_test['test'].train_test_split(test_size=0.3, seed=SEED)

#         train = train_val_test['train']
#         test = val_test_split['test'] # Correctly assign test set
#         val = val_test_split['train'] # Correctly assign validation set

#         dataset = DatasetDict({
#             'train': train,
#             'test': test,
#             'val': val
#         })
#         return dataset

#     # Removed training_model_setup as setup is streamlined in prepare_for_training
#     # def training_model_setup(self):
#     #     ...

#     def get_data_collator(self, response_template: str):
#         # Ensure response_template is correctly identified if needed by collator
#         # For Unsloth, often the standard collator works, or specific formatting is handled elsewhere
#         # Using the standard TRL collator here, adjust if Unsloth requires specifics
#         return DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

#     # Lora setup is now integrated into model loading with FastLanguageModel.get_peft_model
#     # def prepare_lora_model(self):
#     #     ...

#     def train_model(self, max_seq_length=512, num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,gradient_accumulation_steps=4, optim="adamw_8bit", # Changed optimizer to one supported by Unsloth
# eval_strategy="steps",eval_steps=0.2,save_steps=0.2,logging_steps=10,learning_rate=1e-4,fp16=None,bf16=None, # Let Unsloth handle precision
# save_strategy="steps", warmup_ratio=0.1, save_total_limit=2, lr_scheduler_type="linear", # Changed scheduler
# report_to="tensorboard", save_safetensors=True, dataset_kwargs=None): # Simplified dataset_kwargs

#         # Add LoRA adapters using Unsloth's method
#         self.model = FastLanguageModel.get_peft_model(
#             self.model,
#             r = 32, # LoRA rank
#             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                               "gate_proj", "up_proj", "down_proj"], # Standard Llama3 modules
#             lora_alpha = 16,
#             lora_dropout = 0.05,
#             bias = "none",
#             use_gradient_checkpointing = True, # Recommended for memory saving
#             random_state = SEED,
#             max_seq_length = max_seq_length,
#         )

#         response_template = self.tokenizer.eos_token # Or adjust based on dataset formatting
#         collator = self.get_data_collator(response_template)

#         # Use Unsloth compatible SFTConfig parameters
#         sft_config = SFTConfig(
#             output_dir=self.output_dir,
#             dataset_text_field="text",
#             max_seq_length=max_seq_length,
#             num_train_epochs=num_train_epochs,
#             per_device_train_batch_size=per_device_train_batch_size,
#             per_device_eval_batch_size=per_device_eval_batch_size,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             optim=optim,
#             eval_strategy=eval_strategy,
#             eval_steps=eval_steps,
#             save_steps=save_steps,
#             logging_steps=logging_steps,
#             learning_rate=learning_rate,
#             # fp16=fp16, # Unsloth manages precision
#             # bf16=bf16,
#             save_strategy=save_strategy,
#             warmup_ratio=warmup_ratio,
#             save_total_limit=save_total_limit,
#             lr_scheduler_type=lr_scheduler_type,
#             report_to=report_to,
#             save_safetensors=save_safetensors,
#             # dataset_kwargs=dataset_kwargs, # Often not needed or handled differently
#             seed=SEED,
#         )

#         trainer = SFTTrainer(
#             model=self.model,
#             args=sft_config,
#             train_dataset=self.dataset["train"],
#             eval_dataset=self.dataset["val"],
#             tokenizer=self.tokenizer,
#             data_collator=collator,
#         )
#         trainer.train()
#         # Save the LoRA adapter model
#         trainer.model.save_pretrained(self.output_dir) # Save LoRA adapters to output_dir
#         self.tokenizer.save_pretrained(self.output_dir) # Save tokenizer alongside adapters
#         self.clear_memory()
#         print(f"Finished fine-tuning. LoRA adapters saved to {self.output_dir}")

# # Updated function to merge and save the final model using Unsloth
# def convert_and_save_model():
#     # Clear memory before loading the base model for merging
#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()
#     print(f"Loading base model ({BASE_MODEL}) for merging...")
#     # Load the base model again, potentially without 4-bit if merging requires more precision
#     # Check Unsloth documentation for best practices on merging (sometimes done during inference)
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = BASE_MODEL, # Use the same base model used for training
#         max_seq_length = 512,
#         dtype = None, # Or torch.float16
#         load_in_4bit = False, # Load in higher precision for merging if needed
#     )

#     print(f"Loading LoRA adapters from {OUTPUT_DIR}...")
#     # Load the trained LoRA adapters
#     # This step might differ based on how Unsloth handles PeftModel loading
#     # Assuming direct loading works, adjust if necessary
#     try:
#         # Unsloth might automatically handle merging or provide a specific function
#         # If PeftModel loading is standard:
#         model = PeftModel.from_pretrained(model, OUTPUT_DIR)
#         print("Merging LoRA adapters...")
#         model = model.merge_and_unload()
#         print(f"Saving merged model to {NEW_MODEL}...")
#         model.save_pretrained(NEW_MODEL)
#         tokenizer.save_pretrained(NEW_MODEL)
#         print("Finished Merging and Local Saving")

#     except Exception as e:
#         print(f"Could not directly merge. Trying to save adapters only from {OUTPUT_DIR} to {NEW_MODEL}.")
#         print("You might need to load the base model and adapters separately for inference.")
#         # Fallback: If direct merge fails, just ensure adapters are saved (already done by trainer)
#         # Or copy the adapter files if needed
#         # import shutil
#         # shutil.copytree(OUTPUT_DIR, NEW_MODEL, dirs_exist_ok=True)
#         print(f"Adapters are available in {OUTPUT_DIR}. Merged model saving failed with error: {e}")

#     # Memory is cleared at the beginning of the function now
#     # import gc
#     # gc.collect()
#     # torch.cuda.empty_cache()


# def main():
#     # Prepare and train the model using the Unsloth-integrated class
#     trainer = ModelTrainer.prepare_for_training(BASE_MODEL, NEW_MODEL, PAD_TOKEN, OUTPUT_DIR, DATASET_PATH)
#     trainer.train_model()

#     # Explicitly delete trainer and clear memory before merging
#     print("Clearing trainer and CUDA cache before merging...")
#     del trainer
#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()

#     # Attempt to merge and save the model (optional, can use adapters directly)
#     convert_and_save_model() # Uncomment if you need a fully merged model saved

# if __name__ == "__main__":
#     main()


# import random
# import torch
# import numpy as np
# from datasets import DatasetDict, load_dataset
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer
# from trl import SFTConfig, SFTTrainer
# import gc

# # Constants
# SEED = 42
# BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
# NEW_MODEL_NAME = "LLM_Wizard/Qwen2.5-VL-7B-Unsloth"
# OUTPUT_DIR = "outputs"
# DATASET_PATH = "LLM_Wizard/dataset/final_templated_finetuning_data.parquet"
# MAX_SEQ_LENGTH = 2048

# class ModelTrainer:
#     def __init__(self, base_model: str, output_dir: str, dataset_path: str, max_seq_length: int):
#         self.base_model = base_model
#         self.output_dir = output_dir
#         self.dataset_path = dataset_path
#         self.max_seq_length = max_seq_length
#         self.tokenizer = None
#         self.model = None
#         self.dataset = None
#         self.seed_everything(SEED)

#     @staticmethod
#     def seed_everything(seed: int):
#         """Set seed for reproducibility."""
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)

#     @staticmethod
#     def clear_memory():
#         """Clear GPU memory."""
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def load_model_and_tokenizer(self):
#         """Load the model and tokenizer using Unsloth's FastLanguageModel."""
#         print(f"Loading Unsloth model: {self.base_model}")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=self.base_model,
#             max_seq_length=self.max_seq_length,
#             dtype=None,  # Unsloth handles dtype automatically
#             load_in_4bit=True,
#         )

#         # Set padding token to EOS token for consistency
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         self.model, self.tokenizer = model, tokenizer

#     def load_and_prepare_dataset(self):
#         """Load and split the dataset. Assumes data is pre-formatted."""
#         # Load the parquet file
#         dataset = load_dataset("parquet", data_files=self.dataset_path)

#         # Create train/val/test splits
#         train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=SEED)
#         val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=SEED)

#         self.dataset = DatasetDict({
#             'train': train_val_split['train'],
#             'val': val_test_split['train'],
#             'test': val_test_split['test']
#         })
#         print("Dataset splits created:")
#         print(f"Training set size: {len(self.dataset['train'])}")
#         print(f"Validation set size: {len(self.dataset['val'])}")
#         print(f"Test set size: {len(self.dataset['test'])}")

#     def train_model(self):
#         """Configure and run the fine-tuning process on the pre-templated dataset."""
#         # Add LoRA adapters to the model, allowing Unsloth to auto-detect target modules
#         self.model = FastLanguageModel.get_peft_model(
#             self.model,
#             r=32,
#             lora_alpha=32,
#             lora_dropout=0.05,
#             bias="none",
#             use_gradient_checkpointing=True,
#             random_state=SEED,
#         )

#         # SFT Configuration - dataset_text_field must match the column with pre-templated text
#         sft_config = SFTConfig(
#             output_dir=self.output_dir,
#             dataset_text_field="text", # Assumes the parquet file has a "text" column
#             max_seq_length=self.max_seq_length,
#             num_train_epochs=1,
#             per_device_train_batch_size=2,
#             per_device_eval_batch_size=2,
#             gradient_accumulation_steps=4,
#             optim="adamw_8bit",
#             eval_strategy="steps",
#             eval_steps=100,
#             save_steps=100,
#             logging_steps=10,
#             learning_rate=1e-4,
#             warmup_ratio=0.1,
#             save_total_limit=2,
#             lr_scheduler_type="linear",
#             seed=SEED,
#             report_to="tensorboard",
#         )

#         trainer = SFTTrainer(
#             model=self.model,
#             args=sft_config,
#             train_dataset=self.dataset["train"],
#             eval_dataset=self.dataset["val"],
#             tokenizer=self.tokenizer,
#         )

#         print("Starting fine-tuning...")
#         trainer.train()

#         print(f"Finished fine-tuning. LoRA adapters saved to {self.output_dir}")
#         self.clear_memory()

#     def save_and_merge_model(self, final_model_name: str):
#         """Save the final merged model using Unsloth's streamlined method."""
#         print("Merging LoRA adapters and saving the final model...")
#         self.clear_memory()

#         # Reload the base model to merge adapters.
#         # This is memory-efficient and ensures a clean state.
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=self.base_model,
#             max_seq_length=self.max_seq_length,
#             dtype=None,
#             load_in_4bit=True,
#         )
        
#         # Merge the LoRA adapters into the model and save
#         model.save_pretrained_merged(
#             save_directory=final_model_name,
#             tokenizer=tokenizer,
#             save_method="merged_16bit",  # Saves in float16, good for deployment
#         )
#         print(f"Final 16-bit merged model saved to {final_model_name}")

# def main():
#     """Main function to run the fine-tuning workflow."""
#     trainer = ModelTrainer(
#         base_model=BASE_MODEL,
#         output_dir=OUTPUT_DIR,
#         dataset_path=DATASET_PATH,
#         max_seq_length=MAX_SEQ_LENGTH
#     )
    
#     # Load model, tokenizer, and dataset
#     trainer.load_model_and_tokenizer()
#     trainer.load_and_prepare_dataset()
#     trainer.clear_memory()
    
#     # Start the training process
#     trainer.train_model()
    
#     # Save the final, merged model
#     trainer.save_and_merge_model(final_model_name=NEW_MODEL_NAME)
#     print("Workflow complete.")

# if __name__ == "__main__":
#     main()


# import random
# import torch
# import numpy as np
# from datasets import DatasetDict, load_dataset
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer
# from trl import SFTConfig, SFTTrainer
# import gc

# # Constants
# SEED = 42
# BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"
# NEW_MODEL_NAME = "LLM_Wizard/Qwen2.5-VL-7B-Unsloth"
# OUTPUT_DIR = "outputs"
# DATASET_PATH = "LLM_Wizard/dataset/final_templated_finetuning_data.parquet"
# MAX_SEQ_LENGTH = 2048

# class ModelTrainer:
#     def __init__(self, base_model: str, output_dir: str, dataset_path: str, max_seq_length: int):
#         self.base_model = base_model
#         self.output_dir = output_dir
#         self.dataset_path = dataset_path
#         self.max_seq_length = max_seq_length
#         self.tokenizer = None
#         self.model = None
#         self.dataset = None
#         self.seed_everything(SEED)

#     @staticmethod
#     def seed_everything(seed: int):
#         """Set seed for reproducibility."""
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)

#     @staticmethod
#     def clear_memory():
#         """Clear GPU memory."""
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     def load_model_and_tokenizer(self):
#         """Load the model and tokenizer using Unsloth's FastLanguageModel."""
#         print(f"Loading Unsloth model: {self.base_model}")
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=self.base_model,
#             max_seq_length=self.max_seq_length,
#             dtype=None,
#             load_in_4bit=True,
#         )

#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         self.model, self.tokenizer = model, tokenizer

#     def load_and_prepare_dataset(self):
#         """Load and split the dataset."""
#         dataset = load_dataset("parquet", data_files=self.dataset_path)

#         train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=SEED)
#         val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=SEED)

#         self.dataset = DatasetDict({
#             'train': train_val_split['train'],
#             'val': val_test_split['train'],
#             'test': val_test_split['test']
#         })
#         print("Dataset splits created:")
#         print(f"Training set size: {len(self.dataset['train'])}")
#         print(f"Validation set size: {len(self.dataset['val'])}")
#         print(f"Test set size: {len(self.dataset['test'])}")
#         print(f"Test set size: {self.dataset['test']["formatted_chat"]}")

#     @staticmethod
#     def formatting_prompts_func(examples):
#         """
#         Processes a batch of examples.
#         The SFTTrainer expects this function to return a list of strings.
#         'examples' is a dictionary where keys are column names and values are lists.
#         """
#         # The function now correctly returns the list of strings from the 'text' column.
#         return [examples["formatted_chat"]]

#     def train_model(self):
#         """Configure and run the fine-tuning process."""
#         self.model = FastLanguageModel.get_peft_model(
#             self.model,
#             r=32,
#             lora_alpha=32,
#             lora_dropout=0.05,
#             bias="none",
#             use_gradient_checkpointing=True,
#             random_state=SEED,
#         )

#         sft_config = SFTConfig(
#             output_dir=self.output_dir,
#             max_seq_length=self.max_seq_length,
#             num_train_epochs=1,
#             per_device_train_batch_size=2,
#             per_device_eval_batch_size=2,
#             gradient_accumulation_steps=4,
#             optim="adamw_8bit",
#             eval_strategy="steps",
#             eval_steps=100,
#             save_steps=100,
#             logging_steps=10,
#             learning_rate=1e-4,
#             warmup_ratio=0.1,
#             save_total_limit=2,
#             lr_scheduler_type="linear",
#             seed=SEED,
#             report_to="tensorboard",
#             # `dataset_text_field` is removed as `formatting_func` is used.
#         )

#         trainer = SFTTrainer(
#             model=self.model,
#             args=sft_config,
#             train_dataset=self.dataset["train"],
#             eval_dataset=self.dataset["val"],
#             tokenizer=self.tokenizer,
#             formatting_func=self.formatting_prompts_func,
#         )

#         print("Starting fine-tuning...")
#         trainer.train()

#         print(f"Finished fine-tuning. LoRA adapters saved to {self.output_dir}")
#         self.clear_memory()

#     def save_and_merge_model(self, final_model_name: str):
#         """Save the final merged model using Unsloth's streamlined method."""
#         print("Merging LoRA adapters and saving the final model...")
#         self.clear_memory()

#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=self.base_model,
#             max_seq_length=self.max_seq_length,
#             dtype=None,
#             load_in_4bit=True,
#         )
        
#         model.save_pretrained_merged(
#             save_directory=final_model_name,
#             tokenizer=tokenizer,
#             save_method="merged_16bit",
#         )
#         print(f"Final 16-bit merged model saved to {final_model_name}")

# def main():
#     """Main function to run the fine-tuning workflow."""
#     trainer = ModelTrainer(
#         base_model=BASE_MODEL,
#         output_dir=OUTPUT_DIR,
#         dataset_path=DATASET_PATH,
#         max_seq_length=MAX_SEQ_LENGTH
#     )
    
#     trainer.load_model_and_tokenizer()
#     trainer.load_and_prepare_dataset()
#     trainer.clear_memory()
    
#     trainer.train_model()
    
#     trainer.save_and_merge_model(final_model_name=NEW_MODEL_NAME)
#     print("Workflow complete.")

# if __name__ == "__main__":
#     main()

# import random
# import torch
# import numpy as np
# from datasets import DatasetDict, load_dataset
# from unsloth import FastVisionModel, is_bf16_supported
# from unsloth.trainer import UnslothVisionDataCollator
# from transformers import TextStreamer
# from trl import SFTConfig, SFTTrainer
# import gc
# import os
# from typing import Dict, List, Optional, Union
# from enum import Enum

# class TrainingMode(Enum):
#     """Enum to define different training modes for the vision model"""
#     VISION_AND_LANGUAGE = "vision_and_language"
#     LANGUAGE_ONLY = "language_only"
#     VISION_ONLY = "vision_only"
#     ATTENTION_ONLY = "attention_only"
#     MLP_ONLY = "mlp_only"

# # Constants
# SEED = 42
# DEFAULT_MODELS = {
#     "llama_3_2_11b": "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
#     "llama_3_2_90b": "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
#     "pixtral_12b": "unsloth/Pixtral-12B-2409-bnb-4bit",
#     "qwen2_vl_7b": "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
#     "qwen2_vl_72b": "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
#     "qwen2_5_vl_7b": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
#     "llava_1_5_7b": "unsloth/llava-1.5-7b-hf-bnb-4bit",
#     "llava_1_6_mistral_7b": "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
# }

# class VisionModelTrainer:
#     """
#     A comprehensive trainer for vision-language models using Unsloth.
#     Supports flexible training modes to fine-tune different components.
#     """
    
#     def __init__(
#         self,
#         base_model: str,
#         output_dir: str,
#         max_seq_length: int = 2048,
#         load_in_4bit: bool = True,
#         use_gradient_checkpointing: Union[bool, str] = "unsloth"
#     ):
#         self.base_model = base_model
#         self.output_dir = output_dir
#         self.max_seq_length = max_seq_length
#         self.load_in_4bit = load_in_4bit
#         self.use_gradient_checkpointing = use_gradient_checkpointing
        
#         self.model = None
#         self.tokenizer = None
#         self.dataset = None
        
#         self.seed_everything(SEED)
        
#     @staticmethod
#     def seed_everything(seed: int):
#         """Set random seeds for reproducibility"""
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
    
#     @staticmethod
#     def clear_memory():
#         """Clear GPU and system memory"""
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
    
#     def load_model_and_tokenizer(self):
#         """Load the vision model and tokenizer"""
#         print(f"Loading Vision Model: {self.base_model}")
        
#         model, tokenizer = FastVisionModel.from_pretrained(
#             self.base_model,
#             load_in_4bit=self.load_in_4bit,
#             use_gradient_checkpointing=self.use_gradient_checkpointing,
#         )
        
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
            
#         self.model, self.tokenizer = model, tokenizer
#         print("Model and tokenizer loaded successfully!")
    
#     def setup_peft_model(
#         self,
#         training_mode: TrainingMode = TrainingMode.VISION_AND_LANGUAGE,
#         r: int = 16,
#         lora_alpha: int = 16,
#         lora_dropout: float = 0.0,
#         bias: str = "none",
#         use_rslora: bool = False,
#         loftq_config: Optional[Dict] = None,
#         target_modules: Optional[Union[str, List[str]]] = None
#     ):
#         """
#         Setup PEFT model with flexible training configuration.
        
#         Args:
#             training_mode: Which components to fine-tune
#             r: LoRA rank parameter
#             lora_alpha: LoRA alpha parameter
#             lora_dropout: LoRA dropout rate
#             bias: Bias configuration
#             use_rslora: Whether to use rank stabilized LoRA
#             loftq_config: LoftQ configuration
#             target_modules: Specific modules to target
#         """
#         print(f"Setting up PEFT model with training mode: {training_mode.value}")
        
#         # Configure which layers to fine-tune based on training mode
#         if training_mode == TrainingMode.VISION_AND_LANGUAGE:
#             finetune_vision_layers = True
#             finetune_language_layers = True
#             finetune_attention_modules = True
#             finetune_mlp_modules = True
#         elif training_mode == TrainingMode.LANGUAGE_ONLY:
#             finetune_vision_layers = False
#             finetune_language_layers = True
#             finetune_attention_modules = True
#             finetune_mlp_modules = True
#         elif training_mode == TrainingMode.VISION_ONLY:
#             finetune_vision_layers = True
#             finetune_language_layers = False
#             finetune_attention_modules = True
#             finetune_mlp_modules = True
#         elif training_mode == TrainingMode.ATTENTION_ONLY:
#             finetune_vision_layers = True
#             finetune_language_layers = True
#             finetune_attention_modules = True
#             finetune_mlp_modules = False
#         elif training_mode == TrainingMode.MLP_ONLY:
#             finetune_vision_layers = True
#             finetune_language_layers = True
#             finetune_attention_modules = False
#             finetune_mlp_modules = True
        
#         self.model = FastVisionModel.get_peft_model(
#             self.model,
#             finetune_vision_layers=finetune_vision_layers,
#             finetune_language_layers=finetune_language_layers,
#             finetune_attention_modules=finetune_attention_modules,
#             finetune_mlp_modules=finetune_mlp_modules,
#             r=r,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#             bias=bias,
#             random_state=SEED,
#             use_rslora=use_rslora,
#             loftq_config=loftq_config,
#             target_modules=target_modules or "all-linear"
#         )
        
#         print("PEFT model setup completed!")
    
#     def load_dataset_from_parquet(self, dataset_path: str, test_size: float = 0.2):
#         """
#         Load dataset from parquet file and create train/val/test splits.
#         Expected format: messages column with conversation format.
#         """
#         print(f"Loading dataset from: {dataset_path}")
        
#         dataset = load_dataset("parquet", data_files={"train": dataset_path})
        
#         # Create train/validation/test splits
#         train_val_split = dataset['train'].train_test_split(test_size=test_size, seed=SEED)
#         val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=SEED)
        
#         self.dataset = DatasetDict({
#             'train': train_val_split['train'],
#             'val': val_test_split['train'],
#             'test': val_test_split['test']
#         })
        
#         print(f"Dataset loaded with splits:")
#         print(f"  Train: {len(self.dataset['train'])} samples")
#         print(f"  Validation: {len(self.dataset['val'])} samples")
#         print(f"  Test: {len(self.dataset['test'])} samples")
#         print(f"  Columns: {self.dataset['train'].column_names}")
    
#     def load_dataset_from_huggingface(self, dataset_name: str, split: str = "train"):
#         """Load dataset from Hugging Face Hub"""
#         print(f"Loading dataset from Hugging Face: {dataset_name}")
        
#         dataset = load_dataset(dataset_name, split=split)
        
#         # If it's a single split, create train/val/test splits
#         if isinstance(dataset, dict):
#             self.dataset = dataset
#         else:
#             train_val_split = dataset.train_test_split(test_size=0.2, seed=SEED)
#             val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=SEED)
            
#             self.dataset = DatasetDict({
#                 'train': train_val_split['train'],
#                 'val': val_test_split['train'],
#                 'test': val_test_split['test']
#             })
        
#         print("Dataset loaded from Hugging Face!")
    
#     def convert_to_conversation_format(
#         self,
#         text_field: str = "text",
#         image_field: str = "image",
#         instruction: str = "Analyze this image and provide a detailed description."
#     ):
#         """
#         Convert dataset to conversation format for vision tasks.
        
#         Args:
#             text_field: Field containing the target text
#             image_field: Field containing the image
#             instruction: Instruction text for the model
#         """
#         def convert_sample(sample):
#             conversation = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": instruction},
#                         {"type": "image", "image": sample[image_field]}
#                     ]
#                 },
#                 {
#                     "role": "assistant",
#                     "content": [
#                         {"type": "text", "text": sample[text_field]}
#                     ]
#                 }
#             ]
#             return {"messages": conversation}
        
#         print("Converting dataset to conversation format...")
        
#         for split in self.dataset.keys():
#             self.dataset[split] = self.dataset[split].map(convert_sample)
        
#         print("Dataset conversion completed!")
    
#     def train_model(
#         self,
#         num_train_epochs: int = 1,
#         per_device_train_batch_size: int = 2,
#         per_device_eval_batch_size: int = 2,
#         gradient_accumulation_steps: int = 4,
#         learning_rate: float = 2e-4,
#         warmup_steps: int = 5,
#         max_steps: Optional[int] = None,
#         logging_steps: int = 1,
#         eval_steps: int = 100,
#         save_steps: int = 100,
#         optim: str = "adamw_8bit",
#         weight_decay: float = 0.01,
#         lr_scheduler_type: str = "linear",
#         report_to: str = "none",
#         remove_unused_columns: bool = False,
#         dataset_text_field: str = "",
#         dataset_kwargs: Optional[Dict] = None,
#         dataset_num_proc: int = 4
#     ):
#         """Train the vision model with specified configuration"""
        
#         print("Preparing model for training...")
#         FastVisionModel.for_training(self.model)
        
#         # Setup training arguments
#         training_args = SFTConfig(
#             output_dir=self.output_dir,
#             per_device_train_batch_size=per_device_train_batch_size,
#             per_device_eval_batch_size=per_device_eval_batch_size,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             warmup_steps=warmup_steps,
#             max_steps=max_steps,
#             num_train_epochs=num_train_epochs if max_steps is None else None,
#             learning_rate=learning_rate,
#             fp16=not is_bf16_supported(),
#             bf16=is_bf16_supported(),
#             logging_steps=logging_steps,
#             eval_strategy="steps" if self.dataset and "val" in self.dataset else "no",
#             eval_steps=eval_steps,
#             save_steps=save_steps,
#             optim=optim,
#             weight_decay=weight_decay,
#             lr_scheduler_type=lr_scheduler_type,
#             seed=SEED,
#             report_to=report_to,
#             # Vision-specific parameters
#             remove_unused_columns=remove_unused_columns,
#             dataset_text_field=dataset_text_field,
#             dataset_kwargs=dataset_kwargs or {"skip_prepare_dataset": True},
#             dataset_num_proc=dataset_num_proc,
#             max_seq_length=self.max_seq_length,
#         )
        
#         # Setup trainer
#         trainer = SFTTrainer(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
#             train_dataset=self.dataset["train"] if self.dataset else None,
#             eval_dataset=self.dataset["val"] if self.dataset and "val" in self.dataset else None,
#             args=training_args,
#         )
        
#         print("Starting training...")
#         trainer_stats = trainer.train()
        
#         print("Training completed! Saving model...")
#         self.model.save_pretrained(self.output_dir)
#         self.tokenizer.save_pretrained(self.output_dir)
        
#         # Print training statistics
#         print(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
#         print(f"Training completed in {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
        
#         return trainer_stats
    
#     def test_inference(
#         self,
#         test_image=None,
#         instruction: str = "Analyze this image and provide a detailed description.",
#         max_new_tokens: int = 128,
#         temperature: float = 1.5,
#         min_p: float = 0.1
#     ):
#         """Test the model with inference"""
#         print("Testing model inference...")
        
#         FastVisionModel.for_inference(self.model)
        
#         if test_image is None and self.dataset and "test" in self.dataset:
#             # Use first test sample if available
#             test_sample = self.dataset["test"][0]
#             test_image = test_sample.get("image") or test_sample.get("messages", [{}])[0].get("content", [{}])[1].get("image")
        
#         if test_image is None:
#             print("No test image available for inference testing")
#             return
        
#         messages = [
#             {"role": "user", "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": instruction}
#             ]}
#         ]
        
#         input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#         inputs = self.tokenizer(
#             test_image,
#             input_text,
#             add_special_tokens=False,
#             return_tensors="pt",
#         ).to("cuda")
        
#         text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
#         print("Model output:")
#         _ = self.model.generate(
#             **inputs,
#             streamer=text_streamer,
#             max_new_tokens=max_new_tokens,
#             use_cache=True,
#             temperature=temperature,
#             min_p=min_p
#         )
    
#     def save_merged_model(
#         self,
#         save_directory: str,
#         save_method: str = "merged_16bit"
#     ):
#         """Save the final merged model"""
#         print(f"Saving merged model to: {save_directory}")
        
#         if save_method == "merged_16bit":
#             self.model.save_pretrained_merged(save_directory, self.tokenizer)
#         else:
#             # For other formats, you might need different methods
#             self.model.save_pretrained(save_directory)
#             self.tokenizer.save_pretrained(save_directory)
        
#         print("Model saved successfully!")
    
#     def push_to_hub(
#         self,
#         repo_name: str,
#         token: str,
#         save_method: str = "merged_16bit"
#     ):
#         """Push the model to Hugging Face Hub"""
#         print(f"Pushing model to Hub: {repo_name}")
        
#         if save_method == "merged_16bit":
#             self.model.push_to_hub_merged(repo_name, self.tokenizer, token=token)
#         else:
#             self.model.push_to_hub(repo_name, token=token)
#             self.tokenizer.push_to_hub(repo_name, token=token)
        
#         print("Model pushed to Hub successfully!")

# def create_sample_latex_ocr_workflow():
#     """Example workflow for LaTeX OCR fine-tuning"""
    
#     # Initialize trainer
#     trainer = VisionModelTrainer(
#         base_model=DEFAULT_MODELS["qwen2_5_vl_7b"],
#         output_dir="./outputs/latex_ocr_model",
#         max_seq_length=2048
#     )
    
#     # Load model
#     trainer.load_model_and_tokenizer()
    
#     # Setup for language-only training (preserving vision capabilities)
#     trainer.setup_peft_model(
#         training_mode=TrainingMode.LANGUAGE_ONLY,
#         r=16,
#         lora_alpha=16,
#         lora_dropout=0.0
#     )
    
#     # Load LaTeX OCR dataset
#     trainer.load_dataset_from_huggingface("unsloth/LaTeX_OCR")
    
#     # Convert to conversation format
#     trainer.convert_to_conversation_format(
#         text_field="text",
#         image_field="image",
#         instruction="Write the LaTeX representation for this image."
#     )
    
#     # Train the model
#     trainer.train_model(
#         num_train_epochs=1,
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=2e-4,
#         max_steps=30  # For demo purposes
#     )
    
#     # Test inference
#     trainer.test_inference(
#         instruction="Write the LaTeX representation for this image."
#     )
    
#     # Save the model
#     trainer.save_merged_model("./final_models/latex_ocr_merged")
    
#     print("LaTeX OCR fine-tuning workflow completed!")

# def create_sample_combined_workflow():
#     """Example workflow for combined vision+language fine-tuning"""
    
#     trainer = VisionModelTrainer(
#         base_model=DEFAULT_MODELS["llama_3_2_11b"],
#         output_dir="./outputs/combined_model"
#     )
    
#     trainer.load_model_and_tokenizer()
    
#     # Setup for combined training
#     trainer.setup_peft_model(
#         training_mode=TrainingMode.VISION_AND_LANGUAGE,
#         r=32,
#         lora_alpha=32,
#         lora_dropout=0.05
#     )
    
#     # Load your custom dataset
#     # trainer.load_dataset_from_parquet("path/to/your/dataset.parquet")
    
#     # Train the model
#     trainer.train_model(
#         num_train_epochs=1,
#         per_device_train_batch_size=1,  # Reduce for larger models
#         gradient_accumulation_steps=8,
#         learning_rate=1e-4
#     )
    
#     print("Combined vision+language fine-tuning completed!")

# if __name__ == "__main__":
#     # Clear any existing memory
#     VisionModelTrainer.clear_memory()
    
#     # Run sample workflows
#     print("=" * 50)
#     print("Running LaTeX OCR Workflow (Language-only training)")
#     print("=" * 50)
#     create_sample_latex_ocr_workflow()
    
#     # Uncomment to run combined workflow
#     # print("\n" + "=" * 50)
#     # print("Running Combined Vision+Language Workflow")
#     # print("=" * 50)
#     # create_sample_combined_workflow()





# from unsloth import FastVisionModel, UnslothVisionDataCollator
# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import TrainingArguments
# from trl import SFTTrainer, SFTConfig


# class VisionModelFineTuner:
#     """
#     A class to fine-tune Vision Language Models using Unsloth, with options
#     to selectively train the vision encoder, the language model, or both.
#     """
#     def __init__(self, model_name="unsloth/Llama-3.2-11B-Vision-Instruct", max_seq_length=2048, load_in_4bit=True):
#         """
#         Initializes the fine-tuner with model and tokenizer configurations.

#         Args:
#             model_name (str): The name of the model to load from Hugging Face Hub.
#             max_seq_length (int): The maximum sequence length for the model.
#             load_in_4bit (bool): Whether to load the model in 4-bit precision.
#         """
#         self.model_name = model_name
#         self.max_seq_length = max_seq_length
#         self.load_in_4bit = load_in_4bit
#         self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
#         print("Initializing VisionModelFineTuner...")
#         self.model, self.tokenizer = FastVisionModel.from_pretrained(
#             model_name=self.model_name,
#             max_seq_length=self.max_seq_length,
#             dtype=self.dtype,
#             load_in_4bit=self.load_in_4bit,
#         )
#         print("Model and Tokenizer loaded successfully.")

#     def _prepare_text_only_dataset(self, df: pd.DataFrame):
#         """
#         Prepares a text-only conversational dataset for fine-tuning the LLM part.
#         The dataset is expected to have 'conversation_id', 'user', 'context', 
#         and 'character' (assistant) columns.
#         """
#         print("Preparing text-only dataset...")
        
#         # Define a system prompt or instruction
#         instructions = "You are a helpful assistant. Please respond in a clear and concise manner."
        
#         conversations = []
#         for conversation_id, group in df.groupby('conversation_id'):
#             messages = [{"role": "system", "content": instructions}]
#             # Sort by turn order if there's a column for it, otherwise assume order is correct
#             # group = group.sort_values(by='turn_id') 
#             for _, row in group.iterrows():
#                 # Combine user input with context
#                 prompt = f"{row['user']}\n\n[Context]: {row['context']}"
#                 messages.append({"role": "user", "content": prompt.strip()})
#                 messages.append({"role": "assistant", "content": row['character']})
#             conversations.append({"messages": messages})

#         return Dataset.from_list(conversations)

#     def _prepare_vision_dataset(self):
#         """
#         Placeholder function to show how to load a dataset with images.
#         For a real use case, you would load your own dataset with 'messages' and 'image' columns.
#         """
#         print("Loading sample vision dataset...")
#         # This is an example of a dataset format for vision fine-tuning.
#         # It must contain a column with a list of conversation turns and an 'image' column.
#         # For example: from datasets import load_dataset
#         # vision_dataset = load_dataset("HuggingFaceH-WING/DocVQA-sampled", split="train")
#         # You would then format it to have 'messages' and 'image' columns.
#         # For now, we'll return a placeholder to demonstrate the logic.
#         from datasets import load_dataset
#         ds = load_dataset("unsloth/dummy_vision_dataset", split = "train")
#         return ds

#     def run_finetuning(
#         self,
#         dataset: pd.DataFrame,
#         finetune_llm: bool = True,
#         finetune_vision: bool = False,
#         output_dir: str = "vision_model_finetuned",
#         training_args_config: dict = None
#     ):
#         """
#         Executes the full fine-tuning pipeline.

#         Args:
#             dataset (pd.DataFrame): The input DataFrame for training.
#             finetune_llm (bool): If True, fine-tunes the language model layers.
#             finetune_vision (bool): If True, fine-tunes the vision encoder layers.
#             output_dir (str): Directory to save training outputs.
#             training_args_config (dict): A dictionary of TrainingArguments.
#         """
#         if not finetune_llm and not finetune_vision:
#             raise ValueError("At least one of `finetune_llm` or `finetune_vision` must be True.")

#         # 1. Configure PEFT based on user selection
#         print(f"Configuring PEFT... LLM tuning: {finetune_llm}, Vision tuning: {finetune_vision}")
#         self.model = FastVisionModel.get_peft_model(
#             self.model,
#             r=32,
#             lora_alpha=32,
#             lora_dropout=0,
#             bias="none",
#             finetune_language_layers=finetune_llm,
#             finetune_vision_layers=finetune_vision,
#         )

#         # 2. Prepare the dataset
#         if finetune_vision:
#             # For vision or mixed tuning, an image column is expected.
#             # We use the placeholder dataset here. For your case, you'd need to add an image column.
#             formatted_dataset = self._prepare_vision_dataset()
#         else:
#             # For text-only LLM tuning, we use the text preparation function.
#             formatted_dataset = self._prepare_text_only_dataset(dataset)

#         # 3. Set up the Trainer
#         default_training_args = {
#             "per_device_train_batch_size": 2,
#             "gradient_accumulation_steps": 4,
#             "warmup_steps": 10,
#             "max_steps": 50, # Set to a small number for demonstration
#             "learning_rate": 1e-5,
#             "fp16": not torch.cuda.is_bf16_supported(),
#             "bf16": torch.cuda.is_bf16_supported(),
#             "logging_steps": 1,
#             "optim": "adamw_8bit",
#             "weight_decay": 0.01,
#             "seed": 42,
#             "output_dir": output_dir,
#         }
#         if training_args_config:
#             default_training_args.update(training_args_config)
            
#         trainer = SFTTrainer(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             train_dataset=formatted_dataset,
#             data_collator=UnslothVisionDataCollator(), # Unsloth's collator handles both cases
#             args=TrainingArguments(**default_training_args),
#             dataset_text_field="messages", # Unsloth uses 'messages' convention
#         )

#         # 4. Start Training
#         print("Starting fine-tuning...")
#         trainer.train()
#         print("Fine-tuning completed.")

#         # 5. Merge adapters and save the full model
#         self.merge_and_save(save_directory=f"{output_dir}/final_merged_model")

#     def merge_and_save(self, save_directory: str):
#         """Merges LoRA adapters and saves the complete model and tokenizer."""
#         print("Merging LoRA adapters into the base model...")
#         self.model.merge_and_unload()
#         print(f"Saving merged model to {save_directory}")
#         self.model.save_pretrained(save_directory)
#         self.tokenizer.save_pretrained(save_directory)
#         print("Model and tokenizer saved successfully.")


# # ==============================================================================
# # EXAMPLE USAGE
# # ==============================================================================
# if __name__ == '__main__':
#     # --- Scenario: Fine-tuning the LLM for conversational style ---
#     # This matches your described use case.
    
#     # 1. Create a sample dataset that mimics your structure.
#     # We have two conversations, each with multiple turns.
#     data = {
#         'conversation_id': [1, 1, 2, 2],
#         'user': [
#             "What is the capital of France?", 
#             "What is it famous for?",
#             "Who wrote 'To Kill a Mockingbird'?",
#             "What's the main theme?"
#         ],
#         'context': [
#             "The user is asking a basic geography question.", 
#             "Follow-up question about landmarks.",
#             "The user is asking about American literature.",
#             "Follow-up question about the book's themes."
#         ],
#         'character': [ # This is the target assistant response
#             "The capital of France is Paris, of course!",
#             "It's renowned for many things, including the majestic Eiffel Tower, the Louvre Museum, and its romantic ambiance.",
#             "That would be the brilliant Harper Lee.",
#             "The book powerfully explores themes of racial injustice, innocence, and moral growth in the American South."
#         ]
#     }
#     my_text_dataframe = pd.DataFrame(data)

#     # 2. Initialize the fine-tuner
#     # Use a smaller model for faster demonstration if needed, like "unsloth/Phi-3-mini-4k-instruct"
#     # For this example, we stick to the powerful Llama 3.2 Vision model.
#     tuner = VisionModelFineTuner(model_name="unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")

#     # 3. Run the fine-tuning process for the LLM ONLY
#     tuner.run_finetuning(
#         dataset=my_text_dataframe,
#         finetune_llm=True,
#         finetune_vision=False, # Crucially, we set this to False
#         output_dir="LLM_Wizard/Qwen2.5-VL-7B-Unsloth",
#     )

#     print("\n" + "="*50)
#     print("Fine-tuning for LLM style completed.")
#     print(f"The final, merged model is saved in 'llm_style_finetuned/final_merged_model'")
#     print("This model retains its original vision capabilities but has an updated conversational style.")
#     print("="*50 + "\n")

#     # --- To fine-tune vision capabilities, you would do the following: ---
#     # print("To fine-tune vision capabilities, you would need a dataset with an 'image' column.")
#     # print("The call would look like this:")
#     # tuner.run_finetuning(
#     #     dataset=your_vision_dataframe, # This DataFrame must have an image column
#     #     finetune_llm=True, # Often you train both together
#     #     finetune_vision=True,
#     #     output_dir="vision_and_llm_finetuned"
#     # )




# # --- Install dependencies (if not done already) ---
# # !pip install unsloth bitsandbytes accelerate trl
# # !pip install git+https://github.com/huggingface/transformers

# # --- Load the model ---
# from unsloth import FastVisionModel
# from unsloth.chat_templates import get_chat_template

# model, tokenizer = FastVisionModel.from_pretrained(
#     "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",  # Change if using your own HF model
#     load_in_4bit=True,           # Use 4-bit quantization
#     max_seq_length=2048,         # Adjust as needed
#     use_gradient_checkpointing="unsloth"
# )

# # Apply chat template for Qwen2.5
# tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# # --- Prepare PEFT model ---
# model = FastVisionModel.get_peft_model(
#     model,
#     finetune_vision_layers=False,     # Skip vision layers during fine-tuning
#     finetune_language_layers=True,    # Enable LLM fine-tuning
#     finetune_attention_modules=True,
#     finetune_mlp_modules=True,
#     r=16, lora_alpha=16, lora_dropout=0.1,
#     bias="lora_only",
#     use_rslora=True,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#         "embed_tokens", "lm_head"
#     ],
# )

# FastVisionModel.for_training(model)  # Enable Unsloth's training optimizations

# # --- Load and process dataset ---
# import pandas as pd
# from datasets import Dataset
# from collections import defaultdict

# # Load Parquet
# df = pd.read_parquet("LLM_Wizard/dataset/finetuning_input.parquet")

# # Group into conversations
# convos = defaultdict(list)
# for idx, row in df.iterrows():
#     cid = row.get("conversation_id", f"conv_{idx}")
    
#     # Add context once at start of conversation
#     if row.get("context") and not convos[cid]:
#         convos[cid].append({
#             "role": "system",
#             "content": [{"type": "text", "text": row["context"]}]
#         })
    
#     # Add user turn
#     if pd.notna(row.get("user")):
#         convos[cid].append({
#             "role": "user",
#             "content": [{"type": "text", "text": row["user"]}]
#         })
    
#     # Add assistant turn
#     if pd.notna(row.get("character")):
#         convos[cid].append({
#             "role": "assistant",
#             "content": [{"type": "text", "text": row["character"]}]
#         })

# # Convert to HuggingFace dataset
# examples = [{"conversations": conv} for conv in convos.values()]
# dataset = Dataset.from_list(examples)

# # Apply chat template
# def format_fn(batch):
#     texts = []
#     for conv in batch["conversations"]:
#         text = tokenizer.apply_chat_template(
#             conv, tokenize=False, add_generation_prompt=True
#         )
#         texts.append(text)
#     return {"text": texts}

# dataset = dataset.map(format_fn, batched=True)

# # --- Set up trainer ---
# from unsloth.trainer import UnslothVisionDataCollator
# from trl import SFTTrainer, SFTConfig

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     packing=False,  # True if you want to pack multiple samples into one sequence
#     args=SFTConfig(
#         output_dir="LLM_Wizard/qwen2.5-vl-finetuned",
#         max_steps=1000,  # Or use num_train_epochs
#         per_device_train_batch_size=1,  # Adjust as needed
#         gradient_accumulation_steps=4,
#         learning_rate=1e-5,
#         warmup_steps=100,
#         save_steps=500,
#         logging_steps=50,
#         fp16=True,  # Or bf16 if available
#     ),
#     data_collator=UnslothVisionDataCollator(tokenizer=tokenizer)
# )

# # --- Train ---
# trainer.train()

# # --- Save ---
# trainer.model.save_pretrained("LLM_Wizard/qwen2.5-vl-finetuned")
# tokenizer.save_pretrained("LLM_Wizard/qwen2.5-vl-finetuned")




# Install dependencies (if not done)
# !pip install unsloth bitsandbytes accelerate trl datasets
# !pip install git+https://github.com/huggingface/transformers

import pandas as pd
from datasets import Dataset

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported

# 1️⃣ Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    # max_seq_length=2048,
    use_gradient_checkpointing="unsloth"
)

# 2️⃣ Set up LoRA/PEFT — disable vision layers
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Skip vision encoder
    finetune_language_layers=True,     # Tune language model
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_rslora=True,
    loftq_config = None, # And LoftQ
    random_state = 3407,


    # target_modules=[
    #     "q_proj", "k_proj", "v_proj", "o_proj",
    #     "gate_proj", "up_proj", "down_proj",
    #     "embed_tokens", "lm_head"
    # ]
)

# 3️⃣ Prepare dataset
# Assume your parquet has column: "text" (already templated)
from datasets import load_dataset
dataset_path = "LLM_Wizard/dataset/final_templated_finetuning_data.parquet"
dataset = load_dataset("parquet", data_files={dataset_path}, split='train')

# df = pd.read_parquet("LLM_Wizard/dataset/final_templated_finetuning_data.parquet")

# # Create HF dataset
# dataset = Dataset.from_pandas(df[["formatted_chat"]])#[["formatted_chat"]]

print(dataset["text"])

def custom_collator_fn(examples):
    return examples


# 4️⃣ Optional: verify data looks right
# print(dataset[0]["formatted_chat"][:500])  # Check a sample

# 5️⃣ Prepare model for training
FastVisionModel.for_training(model)
from transformers import DataCollatorForLanguageModeling
custom_collator_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 6️⃣ Setup trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field = "text",
    # data_collator=UnslothVisionDataCollator(model, tokenizer),
    # packing=False,  # Consider True if many short samples
    data_collator=custom_collator_fn,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,  # or use num_train_epochs instead
        learning_rate=2e-4,
        fp16= not is_bf16_supported(),
        bf16= is_bf16_supported(),  # or bf16 if your hardware supports
        logging_steps=10,
        # save_steps=500,
        # eval_steps=500,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type = "linear",
        seed=3407,
        output_dir="LLM_Wizard/qwen2.5-vl-finetune",
        # save_total_limit=2,
        report_to= "none",
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = True,
        dataset_text_field="text", ##vision tuning set to ""
        dataset_kwargs = {"skip_prepare_dataset": False},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    )
)

# 7️⃣ Train
trainer.train()

# 8️⃣ Merge LoRA weights into the base model
# model = FastVisionModel.merge_lora(model)

# 9️⃣ Save the merged model
save_dir = "LLM_Wizard/qwen2.5-vl-finetune-merged"
model.save_pretrained_merged(save_dir, tokenizer, save_method = "merged_16bit",)
# model.save_pretrained_merged(save_dir)
# tokenizer.save_pretrained(save_dir)

print(f"✅ Model saved to {save_dir}")
