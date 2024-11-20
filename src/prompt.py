from langchain_core.prompts import (
    ChatPromptTemplate, FewShotChatMessagePromptTemplate
)


few_shot_examples = [
    {"input": "Oh, I just loved waiting an hour for cold food. Truly the highlight of my week!", "output": "1"},
    {"input": "Reasonable prices and generous portions. A solid choice for a casual meal.", "output": "0"},
    {"input": "I had high hopes, but the restaurant failed to deliver. The portions were small, and the flavors weren't great.", "output": "0"},
    {"input": "Five stars for the ambiance—if you enjoy the sound of screaming toddlers and broken AC.", "output": "1"},
    # {"input": "Their 'world-famous' pasta must be famous for being bland. Truly unforgettable...because I still regret eating it.", "output": "1"},
    # {"input": "I had high hopes, but the restaurant failed to deliver. The portions were small, and the flavors weren't great.", "output": "0"},
]

system_prompt = '''
Classify whether the given review is sarcastic or not based solely on its content. 
Respond with "1" if the review contains any indication of sarcasm, or "0" if there is no sarcasm detected. 
Give short and coherent responses, following the output format demonstrated in the provided few-shot examples.
Provide your answer as a single digit (i.e. either "1" or "0") without any preamble or additional explanations.

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
