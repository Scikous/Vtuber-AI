from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments,pipeline
import datasets
from peft import PeftModel, PeftConfig
import re

from transformers import AutoModelForCausalLM

def dialogue_generator(model, tokenizer, comments, prompt_template, test_model="unnamed"):
    outputs = []
    #pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, num_return_sequences=1)
    for comment in comments:
        prompt = prompt_template(comment)
        inputs = tokenizer(prompt, return_tensors="pt")
        results = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=78)
        output = tokenizer.batch_decode(results)[0]
        #result = pipe(prompt)
        #print(output)
        outputs.append(output)#result[0]['generated_text'])


        #print(result[0]['generated_text'], "\n"*2, "#"*80)
   # print(outputs[0])
    # with open(f"{test_model}.txt", "w", encoding="utf-8") as f:
    #     for line in outputs:
    #         f.write(line)
    print("Text generation finished")
    return outputs

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
    pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?=\<\/|\<|im_end\||$)"

    match = re.search(pattern, reply, re.DOTALL)
    reply = match.group(0).strip() if match else "Womp Womp"
    print(reply)
    return reply
    