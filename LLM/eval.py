# The full `train` s2lit and the full `test` split as two distinct datasets.
# Split the dataset
# deal with later

from model_utils import model_loader
import LLM.llm_templates as pt


def dialogue_generator(model, tokenizer, comments, PromptTemplate, test_model="unnamed"):
    outputs = []
    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, num_return_sequences=1)
    for comment in comments:
        prompt = PromptTemplate(comment)
        inputs = tokenizer(prompt, return_tensors="pt")
        results = model.generate(
            input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=120)
        output = tokenizer.batch_decode(results)[0]
        # result = pipe(prompt)
        # print(output)
        outputs.append(output)  # result[0]['generated_text'])


#        print(result[0]['generated_text'], "\n"*2, "#"*80)
 #  print(outputs[0])
    with open(f"{test_model}.txt", "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line)
    print("Text generation finished")
    return outputs


model_names = [  # "unnamedSICUA", "unnamedSICUCA","unnamedSICUEA",
    # "unnamedSICUAC", "unnamedSICUACC", "unnamedSICUACCC",
    # "unnamedSIUAC",
    # "unnamedSIUCA", "unnamedSIUEA",
    # "unnamedSIUA"
    # "unnamedSICUACCT", "unnamedSICUACCTT",
    "unnamedSICUACCTTT"
]

# used by CUA


def prompt_template_CUA(dialogue): return f'''
<|im_start|>context
{instructions_string}<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''
# used by CUAC
def prompt_template_CUAC(dialogue): return f'''
<|im_start|>context:
{instructions_string}<|im_end|>
<|im_start|>user:
{dialogue}<|im_end|>
<|im_start|>John:
'''

# used by "unnamedSICUA","unnamedSICUCA", "unnamedSICUEA" ALSO unnamedSICUACCT, unnamedSICUACCTT


def prompt_template_SICUA(dialogue): return f'''
<|im_start|>system
{instructions_string}<|im_end|>
<|im_start|>context
<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''
# used by "unnamedSICUAC/C/C",
def prompt_template_SICUAC(dialogue): return f'''
<|im_start|>system:
{instructions_string}<|im_end|>
<|im_start|>context:
<|im_end|>
<|im_start|>user:
{dialogue}<|im_end|>
<|im_start|>John:
'''
# used by"unnamedSIUA", unnamed SIUCA, SIUEA ALSO unnamedSICUACCTTT
def prompt_template_SIUA(dialogue): return f'''
<|im_start|>system
{instructions_string}<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''


def prompt_template_SIUAC(dialogue): return f'''
<|im_start|>system:
{instructions_string}<|im_end|>
<|im_start|>user:
{dialogue}<|im_end|>
<|im_start|>John:
'''


with open("characters/character.txt", "r") as f:
    character_info = f.readline()
instructions_string = f"""{character_info}"""
print(instructions_string)

comments = []

try:
    with open("characters/questions.txt", "r") as f:
        for question in f:
            comments.append(question.strip())
except OSError:
    print("Woopsie, file not found, teehee, using default questions")
    comments = ["What is your name?", "Do you like coffee",
                "Are you real?", "How about that edging?"]

print(comments)
PromptTemplate = prompt_template_SIUA

base_models = ["TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
               "TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"]
base_model = base_models[1]

PromptTemplate = pt.PromptTemplate(
    instructions_str=instructions_string, character_name="John")
for test_model in model_names:
    outputs = []
    model, tokenizer = model_loader(base_model, test_model)
    outputs = dialogue_generator(
        model, tokenizer, comments, PromptTemplate.capybaraChatML, test_model)
    print(outputs)
    # model

    # custom text generator for quicker chunking of dialogue, WIP
    # generated_text = ""
    # while len(generated_text) < max_length:
    #     # Generate one or more tokens at each step
    #     new_tokens = model.generate(
    #         input_ids=tokenizer.encode(generated_text, return_tensors="pt"),
    #         max_length=1,  # Generate only 1 token at a time
    #         temperature=temperature
    #     )

    # # Decode the generated token and append to the current text
    # generated_text += tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

    # print(tokenizer.batch_decode(outputs)[0], "\n"*3, "#"*40)
