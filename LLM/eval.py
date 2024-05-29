# The full `train` s2lit and the full `test` split as two distinct datasets.
# Split the dataset
#deal with later

from utils import model_loader
import prompt_templates as pt

def dialogue_generator(model, tokenizer, comments, prompt_template, test_model="unnamed"):
    outputs = []
    #pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200, num_return_sequences=1)
    for comment in comments:
        prompt = prompt_template(comment)
        inputs = tokenizer(prompt, return_tensors="pt")
        results = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=120)
        output = tokenizer.batch_decode(results)[0]
        #result = pipe(prompt)
        #print(output)
        outputs.append(output)#result[0]['generated_text'])


#        print(result[0]['generated_text'], "\n"*2, "#"*80)
 #  print(outputs[0])
    with open(f"{test_model}.txt", "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line)
    print("Text generation finished")
    return outputs


model_names = [#"unnamedSICUA", "unnamedSICUCA","unnamedSICUEA",
                #"unnamedSICUAC", "unnamedSICUACC", "unnamedSICUACCC",
                #"unnamedSIUAC",
                #"unnamedSIUCA", "unnamedSIUEA",
                #"unnamedSIUA"
                #"unnamedSICUACCT", "unnamedSICUACCTT",
                "unnamedSICUACCTTT"
                ]

#used by CUA
prompt_template_CUA= lambda dialogue: f'''
<|im_start|>context
{instructions_string}<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''
#used by CUAC
prompt_template_CUAC= lambda dialogue: f'''
<|im_start|>context:
{instructions_string}<|im_end|>
<|im_start|>user:
{dialogue}<|im_end|>
<|im_start|>John:
'''

#used by "unnamedSICUA","unnamedSICUCA", "unnamedSICUEA" ALSO unnamedSICUACCT, unnamedSICUACCTT
prompt_template_SICUA= lambda dialogue: f'''
<|im_start|>system
{instructions_string}<|im_end|>
<|im_start|>context
<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''
#used by "unnamedSICUAC/C/C",
prompt_template_SICUAC= lambda dialogue: f'''
<|im_start|>system:
{instructions_string}<|im_end|>
<|im_start|>context:
<|im_end|>
<|im_start|>user:
{dialogue}<|im_end|>
<|im_start|>John:
'''
#used by"unnamedSIUA", unnamed SIUCA, SIUEA ALSO unnamedSICUACCTTT
prompt_template_SIUA= lambda dialogue: f'''
<|im_start|>system
{instructions_string}<|im_end|>
<|im_start|>user
{dialogue}<|im_end|>
<|im_start|>John
'''

prompt_template_SIUAC= lambda dialogue: f'''
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
    comments = ["What is your name?", "Do you like coffee", "Are you real?", "How about that edging?"]

print(comments)
prompt_template= prompt_template_SIUA

base_models = ["TheBloke/Mistral-7B-Instruct-v0.2-GPTQ","TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ"]
base_model = base_models[1]

prompt_template = pt.prompt_template(instructions_str=instructions_string, character_name="John")
for test_model in model_names:
    outputs = []
    model, tokenizer = model_loader(base_model, test_model)
    outputs = dialogue_generator(model, tokenizer, comments, prompt_template.capybaraChatML, test_model)
    print(outputs)
    #model



















    ####custom text generator for quicker chunking of dialogue, WIP
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