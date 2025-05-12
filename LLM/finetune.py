import random
import torch
import numpy as np
from datasets import DatasetDict, load_dataset
from unsloth import FastLanguageModel # Added Unsloth import
from transformers import AutoTokenizer # Removed AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    # get_peft_model, # Removed, Unsloth handles this internally
    # prepare_model_for_kbit_training, # Removed, Unsloth handles this internally
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

SEED = 42
PAD_TOKEN = "<|pad|>" # Unsloth might handle padding differently, check documentation if issues arise
BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit" # Using Unsloth's optimized model
NEW_MODEL = "LLM/Llama-3-8B-Test-Unsloth"
OUTPUT_DIR = "LLM/Hermes-Test-Unsloth"
DATASET_PATH = "LLM/dataset/unnamedSIUAC.txt"

class ModelTrainer:
    def __init__(self, base_model: str, new_model: str, pad_token: str, output_dir: str, dataset_path: str):
        self.base_model = base_model
        self.new_model = new_model
        self.pad_token = pad_token # Keep for tokenizer setup, but Unsloth might override
        self.output_dir = output_dir
        self.dataset_path = dataset_path
        self.tokenizer = None
        self.model = None
        self.dataset = None

    @classmethod
    def prepare_for_training(cls, base_model: str, new_model: str, pad_token: str, output_dir: str, dataset_path: str):
        trainer = cls(base_model, new_model, pad_token, output_dir, dataset_path)
        trainer.seed_everything(SEED)
        trainer.dataset = trainer.load_and_split_dataset(dataset_path)
        # Tokenizer and model loading are now combined in load_model_and_tokenizer
        trainer.model, trainer.tokenizer = trainer.load_model_and_tokenizer()
        # Lora preparation is handled by FastLanguageModel internally when adding adapters
        # trainer.prepare_lora_model() # Removed
        return trainer

    def clear_memory(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Combined model and tokenizer loading for Unsloth
    def load_model_and_tokenizer(self):
        print(f"Loading Unsloth model: {self.base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.base_model,
            max_seq_length = 512, # Set max sequence length here
            dtype = None, # None will default to torch.float16
            load_in_4bit = True,
        )
        # Setup padding token if necessary, Unsloth might handle this
        if tokenizer.pad_token is None:
             tokenizer.add_special_tokens({"pad_token": self.pad_token})
        tokenizer.padding_side = "right"
        # No need to resize embeddings manually, Unsloth handles it
        # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        return model, tokenizer

    # Removed setup_tokenizer and load_model as they are combined now
    # def setup_tokenizer(self):
    #     ...
    # def load_model(self):
    #     ...

    def load_and_split_dataset(self, data_file: str):
        '''
        Assumes that the datas has been formatted correctly -- usually done through dataset_creator.py

        Also WIP, reads from a text file currently, should be parquet
        '''
        dataset = load_dataset("text", data_files=data_file)
        test_size = 0.1
        # Ensure consistent splitting
        dataset = dataset['train'].train_test_split(test_size=test_size, seed=SEED)
        train_val_test = dataset
        # Further split the test set into validation and test
        val_test_split = train_val_test['test'].train_test_split(test_size=0.3, seed=SEED)

        train = train_val_test['train']
        test = val_test_split['test'] # Correctly assign test set
        val = val_test_split['train'] # Correctly assign validation set

        dataset = DatasetDict({
            'train': train,
            'test': test,
            'val': val
        })
        return dataset

    # Removed training_model_setup as setup is streamlined in prepare_for_training
    # def training_model_setup(self):
    #     ...

    def get_data_collator(self, response_template: str):
        # Ensure response_template is correctly identified if needed by collator
        # For Unsloth, often the standard collator works, or specific formatting is handled elsewhere
        # Using the standard TRL collator here, adjust if Unsloth requires specifics
        return DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

    # Lora setup is now integrated into model loading with FastLanguageModel.get_peft_model
    # def prepare_lora_model(self):
    #     ...

    def train_model(self, max_seq_length=512, num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2,gradient_accumulation_steps=4, optim="adamw_8bit", # Changed optimizer to one supported by Unsloth
eval_strategy="steps",eval_steps=0.2,save_steps=0.2,logging_steps=10,learning_rate=1e-4,fp16=None,bf16=None, # Let Unsloth handle precision
save_strategy="steps", warmup_ratio=0.1, save_total_limit=2, lr_scheduler_type="linear", # Changed scheduler
report_to="tensorboard", save_safetensors=True, dataset_kwargs=None): # Simplified dataset_kwargs

        # Add LoRA adapters using Unsloth's method
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32, # LoRA rank
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"], # Standard Llama3 modules
            lora_alpha = 16,
            lora_dropout = 0.05,
            bias = "none",
            use_gradient_checkpointing = True, # Recommended for memory saving
            random_state = SEED,
            max_seq_length = max_seq_length,
        )

        response_template = self.tokenizer.eos_token # Or adjust based on dataset formatting
        collator = self.get_data_collator(response_template)

        # Use Unsloth compatible SFTConfig parameters
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            # fp16=fp16, # Unsloth manages precision
            # bf16=bf16,
            save_strategy=save_strategy,
            warmup_ratio=warmup_ratio,
            save_total_limit=save_total_limit,
            lr_scheduler_type=lr_scheduler_type,
            report_to=report_to,
            save_safetensors=save_safetensors,
            # dataset_kwargs=dataset_kwargs, # Often not needed or handled differently
            seed=SEED,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["val"],
            tokenizer=self.tokenizer,
            data_collator=collator,
        )
        trainer.train()
        # Save the LoRA adapter model
        trainer.model.save_pretrained(self.output_dir) # Save LoRA adapters to output_dir
        self.tokenizer.save_pretrained(self.output_dir) # Save tokenizer alongside adapters
        self.clear_memory()
        print(f"Finished fine-tuning. LoRA adapters saved to {self.output_dir}")

# Updated function to merge and save the final model using Unsloth
def convert_and_save_model():
    # Clear memory before loading the base model for merging
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Loading base model ({BASE_MODEL}) for merging...")
    # Load the base model again, potentially without 4-bit if merging requires more precision
    # Check Unsloth documentation for best practices on merging (sometimes done during inference)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL, # Use the same base model used for training
        max_seq_length = 512,
        dtype = None, # Or torch.float16
        load_in_4bit = False, # Load in higher precision for merging if needed
    )

    print(f"Loading LoRA adapters from {OUTPUT_DIR}...")
    # Load the trained LoRA adapters
    # This step might differ based on how Unsloth handles PeftModel loading
    # Assuming direct loading works, adjust if necessary
    try:
        # Unsloth might automatically handle merging or provide a specific function
        # If PeftModel loading is standard:
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
        print("Merging LoRA adapters...")
        model = model.merge_and_unload()
        print(f"Saving merged model to {NEW_MODEL}...")
        model.save_pretrained(NEW_MODEL)
        tokenizer.save_pretrained(NEW_MODEL)
        print("Finished Merging and Local Saving")

    except Exception as e:
        print(f"Could not directly merge. Trying to save adapters only from {OUTPUT_DIR} to {NEW_MODEL}.")
        print("You might need to load the base model and adapters separately for inference.")
        # Fallback: If direct merge fails, just ensure adapters are saved (already done by trainer)
        # Or copy the adapter files if needed
        # import shutil
        # shutil.copytree(OUTPUT_DIR, NEW_MODEL, dirs_exist_ok=True)
        print(f"Adapters are available in {OUTPUT_DIR}. Merged model saving failed with error: {e}")

    # Memory is cleared at the beginning of the function now
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()


def main():
    # Prepare and train the model using the Unsloth-integrated class
    trainer = ModelTrainer.prepare_for_training(BASE_MODEL, NEW_MODEL, PAD_TOKEN, OUTPUT_DIR, DATASET_PATH)
    trainer.train_model()

    # Explicitly delete trainer and clear memory before merging
    print("Clearing trainer and CUDA cache before merging...")
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Attempt to merge and save the model (optional, can use adapters directly)
    convert_and_save_model() # Uncomment if you need a fully merged model saved

if __name__ == "__main__":
    main()

