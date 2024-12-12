# 10701-Project

## Code:
**anomaly_data_filter.py**: Calculate anomaly score in original yelp reviews; Output: yelp_anomaly.csv\
**append_data.py**: Append augmented data to the original Yelp review data set; Output: yelp_appended.csv\
**classifier_and_evl.ipynb**: Train and test XGBoost classfiers via pseudo-labeling and active-learning\
**generate_features_for_pseudo.py**: Generate semantic representation features of reviews that are used in XGBoost classfier model; Output: yelp_pseudo_features.csv\
**llm_probability_huggingface.ipynb**: Using Llama-3.1 as the probability estimator of sarcasm using HuggingFace Package; Output: review_500_huggingface.txt\
**llm_probability_locally.py**: Running binning-based sarcasm extraction using Llama-3.1 using LangChain package; Output: top500_llm_bubbled.txt\
**llm.py**: \
**prompt.py**:

## Dataset:
**yelp_anomaly.csv**: Yelp original dataset with additional columns for anomaly score calculation.\
**yelp_appended.csv**: The combined dataset includes reviews from **yelp.csv** and **yelp_sarcasm_review_500.txt**.\
**yelp_pseudo_features.csv**: The combined data set with semantic representation features.\
**yelp_sarcasm_review_500.txt**: The augmented sarcastic reviews generated using GPT-4o.\
**yelp_sarcasm.json**: The 500 generated sarcastic reviews using GPT-4o.\
**yelp.csv**: The original yelp review dataset.
