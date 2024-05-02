from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
import transformers

model_names = [#"unnamedSICUA", "unnamedSICUCA","unnamedSICUEA",
                #"unnamedSICUAC", "unnamedSICUACC", "unnamedSICUACCC",
                #"unnamedSIUAC",
                #"unnamedSIUCA", "unnamedSIUEA",
                #"unnamedSIUA"
                "unnamedSICUACCT", "unnamedSICUACCTT",
                "unnamedSICUACCTTT"
                ]

def model_preprocess(test_model):
    model_name = ["TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"]
    print(model_name[1])
    model = AutoModelForCausalLM.from_pretrained(model_name[1],
                                                device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                                trust_remote_code=False, # prevents running custom model files on your machine
                                                revision="main") # which version of model to use in repo

    tokenizer = AutoTokenizer.from_pretrained(model_name[1], use_fast=True)


    model.train() # model in training mode (dropout modules are activated)

    # enable gradient check pointing
    model.gradient_checkpointing_enable()

    # enable quantized training
    model = prepare_model_for_kbit_training(model)


    # LoRA config
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # LoRA trainable version of model
    model = get_peft_model(model, config)

    # trainable parameter count
    model.print_trainable_parameters()


    # load dataset
    data = load_dataset("text", data_files=f"dataset/{test_model}.txt")


    #load dataset into train and test
    test_size = 0.1

    split_data = data['train'].train_test_split(test_size=0.1)
    data = DatasetDict({
        'train': split_data['train'],
        'test': split_data['test']
    })
    print(data, data["train"][0])

    # create tokenize function
    def tokenize_function(examples):
        # extract text
        text = examples["text"]

        #tokenize and truncate text
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512
        )

        return tokenized_inputs

    # tokenize training and validation datasets
    tokenized_data = data.map(tokenize_function, batched=True)


    # setting pad token
    tokenizer.pad_token = tokenizer.eos_token
    # data collator
    return model, tokenized_data, tokenizer

for test_model in model_names:
    model, tokenized_data, tokenizer = model_preprocess(test_model)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # hyperparameters
    lr = 2e-4
    batch_size = 4
    num_epochs = 30
    # define training arguments
    training_args = transformers.TrainingArguments(
        output_dir= test_model+"-7B-GPTQ",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        bf16=True,
        optim="paged_adamw_8bit",

    )

    # configure trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        args=training_args,
        data_collator=data_collator
    )


    # train model
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # renable warnings
    model.config.use_cache = True

    # Save trained model
    trainer.model.save_pretrained(test_model)
    tokenizer.save_pretrained(test_model)

    loss = trainer.state.log_history
    #print(loss, "\n\n", loss[-1])
    data = f"{test_model} Final Epoch Loss: {loss[-3]['loss']} Train Loss: {loss[-1]['train_loss']}\n"
    with open(f"model_losses.txt", "a", encoding="utf-8") as f:
        f.write(data)
    print("Finished model:", test_model)