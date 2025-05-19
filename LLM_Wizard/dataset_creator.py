import sys

sys.path.insert(0, './LLM')
import pandas as pd
from datasets import load_dataset, DatasetDict
from model_utils import LLMUtils # character_loader
from llm_templates import DataTemplate as dt
from textwrap import dedent
#apply a dataset template from DataTemplate in llm_templates.py
def apply_template(row, template):
    return template(
        user_str=row['user'],
        context_str=row['context'],
        character_str=row['character']
    )

def format_example(row: dict):
    # print('\n',row["context"],'\n', row["user"],'\n',row["character"])
    prompt = dedent(
        f"""
    {row["user"]}

    Information:

    ```
    {row["context"]}
    ```
    """
    )
    messages = [
        {
            "role": "system",
            "content": instructions,
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["character"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


#convert csv data to a dataframe to both a extract expected data AND to convert to .parquet later
def csv_to_pandas(csv_path, data_template):
    df = pd.read_csv(csv_path)
    df = df.fillna('')
    # reformat data to follow a template
    # df['formatted_data'] = df.apply(
    #     apply_template, template=data_template, axis=1)
    print(f"following is: {df['user'][0]}endedhere")
    df['formatted_data'] = df.apply(
    format_example, axis=1)
    print(df['formatted_data'][0])
    # print(df['formatted_data'][0])s
    # print(df[['formatted_data']])
    return df[['formatted_data']]

#extracted data from a .csv file to pandas is written to a .parquet file for smaller file size and speed
def pandas_to_parquet(data, parquet_output_path):
    data.to_parquet(parquet_output_path, engine="pyarrow")


def main():
    parquet_output_path = "LLM/dataset/test.parquet"
    csv_path = "LLM/dataset/test.csv"

    # use custom instructions, user and character names if provided
    
    data_template = dt(instructions_str=instructions, user_name=user_name, character_name=character_name)
    df_data = csv_to_pandas(csv_path, data_template=data_template.capybaraChatML)
    pandas_to_parquet(df_data, parquet_output_path)
    # print(df_data[200], len(df_data))
    # print(data, data["train"][0])
if __name__ == "__main__":
    from transformers import AutoTokenizer
    model = "LLM/CapybaraHermes-2.5-Mistral-7B-GPTQ"#"unsloth/Meta-Llama-3.1-8B"#"LLM/Llama-3-8B-Test" #'LLM/Meta-Llama-3.1-8B/'

    tokenizer = AutoTokenizer.from_pretrained(model)

    print(tokenizer.bos_token, tokenizer.eos_token)

    instructions, user_name, character_name = LLMUtils.load_character(character_info_json='LLM/characters/character.json')
    main()
    data = load_dataset("parquet", data_files="LLM/dataset/john_smith.parquet")
    # print(data['train'][0]['formatted_data'])
    # with open('txd.txt','w') as f:
    #     f.write(data['train'][0]['formatted_data'])



# chat1 = [
#     {"role": "user", "content": "Which is bigger, the moon or the sun?"},
#     {"role": "context", "content": "Which is VBSS, the moon or the sun?"},
#     {"role": "assistant", "content": "The sun."}
# ]
# chat2 = [
#     {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
#     {"role": "assistant", "content": "A bacterium."}
# ]

# dataset = Dataset.from_dict({"chat": [chat1, chat2]})
# dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
# print(dataset['formatted_chat'][0])