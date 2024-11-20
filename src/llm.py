import csv
import json
from pathlib import Path

from tqdm import tqdm

from langchain_ollama import ChatOllama
from prompt import create_prompt


def get_llm_response(query, model):
    prompt = create_prompt()
    chain = prompt | model

    response = chain.invoke({"input": query}).content

    return response

def main():
    LLM_NAME = "llama3.1:8b"
    model = ChatOllama(model=LLM_NAME, temperature=0)

    # read the CSV file and populate the dictionary
    anomaly_data_path = "dataset/yelp_anomaly.csv"
    sarcasm_data = []
    sarcasm_data_path = "dataset/yelp_sarcasm.json"
    Path(sarcasm_data_path).parent.mkdir(parents=True, exist_ok=True)
    with open(anomaly_data_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)  # automatically uses the first line as keys
        with open(sarcasm_data_path, 'w', encoding='utf-8') as json_file:
            for row in reader:
                if float(row['anomaly_score']) >= 1/3:
                    sarcasm_data.append(row)
                    json.dump(row, json_file)
                    json_file.write('\n')

    # write system outputs
    system_output = []
    output_file = "output/llm_sarcasm_labels.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    # label test outputs for comparison and analysis
    for sample in tqdm(sarcasm_data):
        query = sample['text']
        response = get_llm_response(query=query, model=model)
        system_output.append(response)

    with open(output_file, 'w') as file:
        for output in system_output:
            file.write(f"{output}\n")


if __name__ == "__main__":
    main()
