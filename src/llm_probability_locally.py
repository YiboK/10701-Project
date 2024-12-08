from pathlib import Path

import pandas as pd
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)

few_shot_examples = [
    {"input": "Oh, I just loved waiting an hour for cold food. Truly the highlight of my week!", "output": "0.95"},
    {"input": "Reasonable prices and generous portions. A solid choice for a casual meal.", "output": "0.01"},
    {"input": "I had high hopes, but the restaurant failed to deliver. The portions were small, and the flavors weren't great.", "output": "0.15"},
    {"input": "Five stars for the ambiance—if you enjoy the sound of screaming toddlers and broken AC.", "output": "0.90"},
]

system_prompt = '''
You are a sentiment analysis expert specializing in detecting sarcasm in text. 
Your goal is to classify whether the given review is sarcastic or not based solely on its content.
Give short and coherent responses, following the output format demonstrated in the provided few-shot examples.
Provide your answer as a probability score (0-1) for how likely you believe the review is sarcastic, no explanation needed. 
'''

def create_prompt():
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        ),
        examples=few_shot_examples,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return prompt

def get_llm_response(query, model):
    prompt = create_prompt()
    chain = prompt | model

    response = chain.invoke({"input": query}).content

    return response

def main():
    LLM_NAME = "llama3.1:8b"
    model = ChatOllama(model=LLM_NAME, temperature=0)

    # Read the input text file and convert to DataFrame
    input_path = "dataset/yelp_sarcasm_review_500.txt"
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    df = pd.DataFrame(lines, columns=['Review'])


    output_file = "output/review_500_few_locally.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    if 'Review' not in df.columns:
        raise ValueError("The input file must contain a 'Review' column or text data in each line.")

    system_output = []

    # Process each review and get the sarcasm probability from LLM
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Reviews"):
        query = row['Review']
        if not query or query.strip() == "":
            system_output.append(None)
            continue
        
        response = get_llm_response(query=query, model=model)
        
        if response:
            try:
                probability = float(response.strip())
            except ValueError:
                print(f"Invalid response for review: {query}. Response: {response}")
                probability = None
        else:
            probability = None

        system_output.append(probability)
    df['Probability'] = system_output

    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
