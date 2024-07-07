# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset, DatasetDict
# import transformers


# model_names = [#"unnamedSICUA", "unnamedSICUCA","unnamedSICUEA",
#                 #"unnamedSICUAC", "unnamedSICUACC", "unnamedSICUACCC",
#                 #"unnamedSIUAC",
#                 #"unnamedSIUCA", "unnamedSIUEA",
#                 #"unnamedSIUA"
#                 "unnamedSICUACCT", "unnamedSICUACCTT",
#                 "unnamedSICUACCTTT"
#                 ]

# def model_preprocess(test_model):
#     model_name = ["TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"]
#     print(model_name[1])
#     model = AutoModelForCausalLM.from_pretrained(model_name[1],
#                                                 device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
#                                                 trust_remote_code=False, # prevents running custom model files on your machine
#                                                 revision="main") # which version of model to use in repo

#     tokenizer = AutoTokenizer.from_pretrained(model_name[1], use_fast=True)


#     model.train() # model in training mode (dropout modules are activated)

#     # enable gradient check pointing
#     model.gradient_checkpointing_enable()

#     # enable quantized training
#     model = prepare_model_for_kbit_training(model)


#     # LoRA config
#     config = LoraConfig(
#         r=8,
#         lora_alpha=32,
#         target_modules=["q_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )

#     # LoRA trainable version of model
#     model = get_peft_model(model, config)

#     # trainable parameter count
#     model.print_trainable_parameters()


#     # load dataset
#     data = load_dataset("text", data_files=f"dataset/{test_model}.txt")


#     #load dataset into train and test
#     test_size = 0.1

#     split_data = data['train'].train_test_split(test_size=0.1)
#     data = DatasetDict({
#         'train': split_data['train'],
#         'test': split_data['test']
#     })
#     print(data, data["train"][0])

#     # create tokenize function
#     def tokenize_function(examples):
#         # extract text
#         text = examples["text"]

#         #tokenize and truncate text
#         tokenizer.truncation_side = "left"
#         tokenized_inputs = tokenizer(
#             text,
#             return_tensors="np",
#             truncation=True,
#             max_length=512
#         )

#         return tokenized_inputs

#     # tokenize training and validation datasets
#     tokenized_data = data.map(tokenize_function, batched=True)


#     # setting pad token
#     tokenizer.pad_token = tokenizer.eos_token
#     # data collator
#     return model, tokenized_data, tokenizer

# for test_model in model_names:
#     model, tokenized_data, tokenizer = model_preprocess(test_model)

#     data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

#     # hyperparameters
#     lr = 2e-4
#     batch_size = 4
#     num_epochs = 30
#     # define training arguments
#     training_args = transformers.TrainingArguments(
#         output_dir= test_model+"-7B-GPTQ",
#         learning_rate=lr,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         num_train_epochs=num_epochs,
#         weight_decay=0.01,
#         logging_strategy="epoch",
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         gradient_accumulation_steps=4,
#         warmup_steps=2,
#         bf16=True,
#         optim="paged_adamw_8bit",

#     )

#     # configure trainer
#     trainer = transformers.Trainer(
#         model=model,
#         train_dataset=tokenized_data["train"],
#         eval_dataset=tokenized_data["test"],
#         args=training_args,
#         data_collator=data_collator
#     )


#     # train model
#     model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
#     trainer.train()

#     # renable warnings
#     model.config.use_cache = True

#     # Save trained model
#     trainer.model.save_pretrained(test_model)
#     tokenizer.save_pretrained(test_model)

#     loss = trainer.state.log_history
#     #print(loss, "\n\n", loss[-1])
#     data = f"{test_model} Final Epoch Loss: {loss[-3]['loss']} Train Loss: {loss[-1]['train_loss']}\n"
#     with open(f"model_losses.txt", "a", encoding="utf-8") as f:
#         f.write(data)
#     print("Finished model:", test_model)














###################################################
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the model and tokenizer
base_model_name = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load your dataset
dataset = load_dataset("text", data_files=f"LLM/dataset/unnamedSIUAC.txt")#load_dataset('your_dataset_name')  # Replace with your dataset
#     data = load_dataset("text", data_files=f"dataset/{test_model}.txt")
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
#######################################################################################################
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
# print(tokenized_dataset, dataset)

# # Create data collator
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     overwrite_output_dir=True,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     save_steps=10_000,
#     save_total_limit=2,
#     prediction_loss_only=True,
#     fp16=True
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_dataset['train'],
#     eval_dataset=tokenized_dataset['validation']
# )

# # Fine-tune the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')
##########################################################################################

import random
from textwrap import dedent
from typing import Dict, List

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
# from colored import Back, Fore, Style
from datasets import Dataset, load_dataset, DatasetDict

from matplotlib.ticker import PercentFormatter
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

SEED = 42


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


seed_everything(SEED)
PAD_TOKEN = "<|pad|>"
MODEL_NAME = "NousResearch/Hermes-2-Theta-Llama-3-8B"
NEW_MODEL = "Llama-3-8B-Test"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"
print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    #     attn_implementation="flash_attention_2",
    #     attn_implementation="sdpa",
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

# print(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)

dataset = load_dataset("text", data_files=f"LLM/dataset/unnamedSIUAC.txt")#load_dataset('your_dataset_name')  # Replace with your dataset

#  load dataset into train and test
test_size = 0.1

train_val_test = dataset['train'].train_test_split(test_size=0.1)
val_test = train_val_test['test'].train_test_split(test_size=0.3)

train = train_val_test['train']
test = val_test['train']
val = train_val_test['test']

# print(train, test, val)

dataset = DatasetDict({
    'train': train,
    'test': test,
    'val': val
})

# print(dataset, dataset["train"][0])

response_template = "<|im_end|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

examples = [dataset["train"][0]["text"]]
encodings = [tokenizer(e) for e in examples]

dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)



lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
     

OUTPUT_DIR = "Hermes-Test"

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_seq_length=512,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # or bf16=True,
    save_strategy="steps",
    warmup_ratio=0.1,
    save_total_limit=2,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    save_safetensors=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False,  # No need to add additional separator token
    },
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=tokenizer,
    data_collator=collator,
)
trainer.train()


    # print(data, data["train"][0])

# rows = []
# for item in dataset["train"]:
#     rows.append(
#         {
#             "question": item["question"],
#             "context": item["context"],
#             "answer": item["answer"],
#         }
#     )
# df = pd.DataFrame(rows)
# print(df.head())