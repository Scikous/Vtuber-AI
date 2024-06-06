import pandas as pd
from datasets import load_dataset, DatasetDict
from model_utils import character_loader  # character_loader
from llm_templates import DataTemplate as dt

#apply a dataset 
def apply_template(row, template):
    return template(
        user_str=row['user'],
        context_str=row['context'],
        character_str=row['character']
    )


def pandas_to_parquet(data, parquet_output_path):
    data.to_parquet(parquet_output_path, engine="pyarrow")
    data = load_dataset("parquet", data_files=parquet_output_path)
    print(data['train'][0]['formatted_data'])


def csv_to_parquet(csv_path, parquet_output_path, data_template):
    df = pd.read_csv(csv_path)

    # reformat data to follow a template
    df['formatted_data'] = df.apply(
        apply_template, template=data_template, axis=1)
    # print(df['formatted_data'][0])
    pandas_to_parquet(df[['formatted_data']], parquet_output_path)


def main():
    pq_path = "LLM/dataset/template.parquet"
    csv_path = "LLM/dataset/template.csv"
    # use custom instructions, user and character names if provided
    instructions, user_name, character_name = character_loader(character_info_json='LLM/characters/character.json')
    data_template = dt(instructions_str=instructions, user_name=user_name, character_name=character_name)
    csv_to_parquet(csv_path, pq_path, data_template= data_template.capybaraChatML)

    # print(data, data["train"][0])
if __name__ == "__main__":
    main()
