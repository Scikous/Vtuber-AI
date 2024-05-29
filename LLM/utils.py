from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments,pipeline
import datasets
from peft import PeftModel, PeftConfig
import re
import numpy as np

def get_rand_token_len(min_tokens=15, max_tokens=70):
    tokens = np.arange(min_tokens, max_tokens)
    token_weights = np.linspace(start=1.0, stop=0.05, num=max_tokens-min_tokens)
    token_weights /= np.sum(token_weights)
    token_len = np.random.choice(tokens, p=token_weights)
    return token_len

def dialogue_generator(model, tokenizer, comment, prompt_template):
    #pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, num_return_sequences=1)
    max_new_tokens = get_rand_token_len()
    print(max_new_tokens)
    prompt = prompt_template(comment)
    inputs = tokenizer(prompt, return_tensors="pt")

    results = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
    output = tokenizer.batch_decode(results)[0]
    #result = pipe(prompt)
    print("Text generation finished")
    return output

def model_loader(base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
    model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                               # offload="/offload",
                                                revision="main")

    if custom_model_name:
        print(custom_model_name)
        config = PeftConfig.from_pretrained(custom_model_name)
        model = PeftModel.from_pretrained(model, custom_model_name, offload_folder="./offload")

    model.eval()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    return model, tokenizer

def character_reply_cleaner(reply):
    print(reply)
    pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?=\<\/|\<|im_end\||$)"

    match = re.search(pattern, reply, re.DOTALL)
    reply = match.group(0).strip() if match else "Womp Womp"
    print(reply)
    return reply

    