import pandas as pd
from datasets import load_dataset, DatasetDict
from model_utils import LLMUtils # character_loader
from llm_templates import DataTemplate as dt

#apply a dataset template from DataTemplate in llm_templates.py
def apply_template(row, template):
    return template(
        user_str=row['user'],
        context_str=row['context'],
        character_str=row['character']
    )

#extracted data from a .csv file to pandas is written to a .parquet file for smaller file size and speed
def pandas_to_parquet(data, parquet_output_path):
    data.to_parquet(parquet_output_path, engine="pyarrow")
    data = load_dataset("parquet", data_files=parquet_output_path)
    print(data['train'][0]['formatted_data'])

#convert csv data to a dataframe to both a extract expected data AND to convert to .parquet later
def csv_to_pandas(csv_path, data_template):
    df = pd.read_csv(csv_path)
    # reformat data to follow a template
    df['formatted_data'] = df.apply(
        apply_template, template=data_template, axis=1)
    # print(df['formatted_data'][0])s
    return df['formatted_data']

def main():
    parquet_output_path = "LLM/dataset/template.parquet"
    csv_path = "LLM/dataset/template.csv"

    # use custom instructions, user and character names if provided
    instructions, user_name, character_name = LLMUtils.character_loader(character_info_json='LLM/characters/character.json')
    
    data_template = dt(instructions_str=instructions, user_name=user_name, character_name=character_name)
    df_data = csv_to_pandas(csv_path, data_template=data_template.capybaraChatML)
    pandas_to_parquet(df_data[['formatted_data']], parquet_output_path)

    # print(data, data["train"][0])
if __name__ == "__main__":
    main()
