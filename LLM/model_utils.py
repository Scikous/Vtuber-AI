import json
import numpy as np
import re
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments, pipeline
# import datasets
# from difflib import SequenceMatcher

# responses are shorter or longer to create a more natural way of responding
def get_rand_token_len(min_tokens=15, max_tokens=100, input_len=0):
    """
    WIP
    given an input (STT/Comment), the potential response length should have a higher chance of being longer.  
    """
    # Adjust max tokens based on input length to avoid cutting off mid-thought
    adjusted_max_tokens = max(min_tokens, max_tokens - input_len)
    print(adjusted_max_tokens)
    tokens = np.arange(min_tokens, adjusted_max_tokens)
    token_weights = np.linspace(
        start=1.0, stop=0.05, num=adjusted_max_tokens-min_tokens)
    token_weights /= np.sum(token_weights)
    token_len = np.random.choice(tokens, p=token_weights)
    # print("TOKEENSS", token_len, tokens)
    return token_len


def sentence_reducer(output_clean):
    # Find the last occurrence of a sentence stopper
    match = re.search(r'[.!?](?!.*[.!?])', output_clean)
    if match:
        # Position of the last sentence stopper
        pos = match.end()
        print("I AM EHEREREE"*10, pos)
        # Truncate the text at the position after the last stopper
        output_clean = output_clean[:pos].strip()
    return output_clean

# generates character's response to a given input (TTS/Comment)


def dialogue_generator(model, tokenizer, comment, PromptTemplate):

    # generated text may end prematurely, this should help avoid that
    def is_incomplete_sentence(text):
        # reply will end on not ., !, ? clean it beforehand then check
        return text.strip()[-1] not in {'.', '!', '?'}

    max_attempts = 7  # Set a limit to avoid infinite loops
    # continuation_prompt = "Continue from where the text within context left off. It is IMPERATIVE that you do NOT repeat anything already mentioned in the context. YOU simply just continue the previous without copying it."

    prompt = PromptTemplate(user_str=comment)
    comment_tokenized = tokenizer(comment, return_tensors="pt")
    inputs = tokenizer(prompt, return_tensors="pt")

    generated_text = ""
    # gen_texts = []
    for attempt in range(max_attempts):
        # print(f"WOMPWOM: {len(comment_tokenized['input_ids'][0])}")
        max_new_tokens = get_rand_token_len(
            input_len=len(comment_tokenized["input_ids"][0]))
        results = model.generate(input_ids=inputs["input_ids"].to(
            "cuda"), max_new_tokens=max_new_tokens)
        output = tokenizer.batch_decode(results)[0]
        print(output)
        # only add new unique responses to final output
        output_clean = character_reply_cleaner(output).lower()
        generated_text = output_clean
        # if attempt == 0:
        #     generated_text = output_clean
        #     gen_texts.append(generated_text)
        # else:
        #     similarity = SequenceMatcher(None, gen_texts[-1], output_clean).ratio()
        #     if similarity >= 0.4:
        #         new_text_incomplete = is_incomplete_sentence(output_clean)
        #         if not new_text_incomplete:
        #             gen_texts[-1] = output_clean
        #             generated_text = " ".join(gen_texts)
        #             print("*"*30,"\n\n",generated_text, "\n\n", "*"*30)
        #     else:
        #         generated_text = generated_text + ' ' + output_clean #add space after each generated response #character_reply_cleaner(copy.copy(output)).lower() #gen-text only needs actual response(s)
        #         gen_texts.append(output_clean)
        # print("WHEHEEE"*15, generated_text)
        if not is_incomplete_sentence(generated_text) or attempt == max_attempts:
            break  # Break if the sentence is complete or max attempts reached

        # # Prepare for continuation
        # prompt = PromptTemplate(user_str=continuation_prompt, context_str=generated_text)
        # #print(prompt)
        # inputs = tokenizer(prompt, return_tensors="pt")
    # print(f"\n\n\nDKSDOSOJ{generated_text}\n\n\n")

    print("Text generation finished")
    return generated_text


def character_reply_cleaner(reply):
    # print("w"*30, '\n\n', reply, '\n\n',"w"*30, '\n\n')
    try:
        pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?=\<\/|\<|im_end\||$)"

        match = re.search(pattern, reply, re.DOTALL)
        clean_reply = match.group(0).strip() if match else "Womp Womp"
        if clean_reply == "Womp Womp":
            raise ValueError(" Womp Womp")
        print("hmmm")
    except ValueError:  # not sure if getting triggered ever.
        pattern = r"(?<=<\|im_start\|> John\n)\s*(.+?)(?:\n|$)"
        match = re.search(pattern, reply, re.DOTALL)
        clean_reply = match.group(1).strip() if match else "Womp Womp sequel"
        print('ffff', clean_reply, 'ffffff')
    clean_reply = sentence_reducer(clean_reply)
    return clean_reply


def model_loader(base_model_name="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", custom_model_name=""):
    model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 # offload="LLM/offload",
                                                 revision="main")

    if custom_model_name:
        print(custom_model_name)
        config = PeftConfig.from_pretrained(custom_model_name)
        model = PeftModel.from_pretrained(
            model, custom_model_name, offload_folder="LLM/offload")

    model.eval()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    return model, tokenizer


def character_loader(character_info_json=""):
    if character_info_json:
        with open(character_info_json, 'r') as character:
            character_info = json.load(character)
            instructions, user_name, character_name = character_info[
                "instructions"], character_info["user_name"], character_info["character_name"]
    else:
        instructions, user_name, character_name = "", "user", "assistant"
    return instructions, user_name, character_name

