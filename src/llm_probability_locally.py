from pathlib import Path

import pandas as pd
from tqdm import tqdm

from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)

# This code took 29 min 02 s to run on a 16GB RAM, Ultra 9 185H, and RTX 4070 laptop.
# A local LLM model is needed to run this code. The model can be downloaded from the following link: https://ollama.com/

few_shot_examples = [
    {"input": """["Oh, I just loved waiting an hour for cold food. Truly the highlight of my week!", 
    "Reasonable prices and generous portions. A solid choice for a casual meal.",
    "I had high hopes, but the restaurant failed to deliver. The portions were small, and the flavors weren't great.",
    "Five stars for the ambiance—if you enjoy the sound of screaming toddlers and broken AC."]
    """, 
    "output": "Oh, I just loved waiting an hour for cold food. Truly the highlight of my week!"}
]

system_prompt = '''
You are a sarcasm detection expert. 
Your goal is to select one and only one review from a list of reviews that you believe is most likely to be sarcastic.
You must strictly follow these rules:
1. **Return only the sarcastic review**. No extra explanation, no analysis, no context, no additional text. 
2. Your output **must be exactly one sentence**, and it must be one of the input sentences. Do not create, modify, or explain anything.
3. **Strictly follow the format of the few-shot examples**. Do not include anything other than the selected sentence. 
4. If you do not understand or cannot decide, **still choose one sentence** from the list. Do not leave the output empty.
5. **Do not use quotes, labels, or any extra punctuation**. Just copy and return the sarcastic review as-is.

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
    model = ChatOllama(model=LLM_NAME, temperature=0, num_ctx=6148)

    # Read the input text file and convert to DataFrame
    df = pd.read_csv("dataset/yelp_appended.csv")
    df = df[["text"]].sample(frac=1).reset_index(drop=True)

    input_path = "dataset/yelp_sarcasm_review_500.txt"
    with open(input_path, 'r', encoding='utf-8') as file:
        augmented = file.readlines()

    augmented = set([line.strip() for line in augmented])

    output_file = "output/top500_llm_bubbled.txt"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    system_output = []
    is_generated = []

    num_sentences_per_group = 21
    grouped_data = df['text'].groupby(df.index // num_sentences_per_group).apply(list)

    for index, group in tqdm(grouped_data.items(), total=len(grouped_data), desc="Processing Review Groups"):
        query = str(group)
        
        if not query or query.strip() == "":
            system_output.append(None)
            is_generated.append(0)
            continue
        
        try:
            response = get_llm_response(query=query, model=model)
            if response:
                choice = response.strip()
            else:
                choice = None
        except ValueError:
            print(f"Invalid response for review: {query}. Response: {response}")
            choice = None

        is_in_augmented = 1 if choice in augmented else 0
        system_output.append(choice)
        is_generated.append(is_in_augmented)

    result_df = pd.DataFrame({
        'is_in_augmented': is_generated,
        'text': system_output
    })

    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")


if __name__ == "__main__":
    main()