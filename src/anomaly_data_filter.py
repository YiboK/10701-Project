# built-in python library
import os

# third-parth library
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# self import

# device selection
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # for Linux & Windows
device = torch.device('mps' if torch.mps.is_available() else 'cpu') # for OSX


def compute_sentiment_score_1(model_name, texts, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    all_scores = list()
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts,
                           return_tensors='pt',
                           truncation=True,
                           padding=True,
                           max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits
        probs = softmax(scores, dim=1)
        star_ratings = torch.arange(1, 6).float().to(device)
        batch_scores = (probs * star_ratings).sum(dim=1)
        normalized_scores = (batch_scores - 1) / 4
        all_scores.extend(normalized_scores.cpu().numpy())
    return all_scores


def compute_sentiment_score_2(model_name, texts, batch_size=32):
    all_scores = list()
    sentiment_pipeline = pipeline('sentiment-analysis',
                                  model=model_name,
                                  tokenizer=model_name,
                                  device=device,
                                  truncation=True,
                                  padding=True,
                                  max_length=512)
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size].tolist()
        results = sentiment_pipeline(batch_texts)
        for result in results:
            if result['label'] == 'POSITIVE':
                score = result['score']
            else:
                score = 1 - result['score']
            all_scores.append(score)
    return all_scores


def main():
    data = pd.read_csv("dataset/yelp.csv")

    print(data.info())
    print(data.head())

    texts = data['text']

    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    sentiment_score_1 = compute_sentiment_score_1(model_name, texts)
    data['sentiment_score_1'] = sentiment_score_1

    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    sentiment_score_2 = compute_sentiment_score_2(model_name, texts)
    data['sentiment_score_2'] = sentiment_score_2

    print(data[["text", "stars", "sentiment_score_1", "sentiment_score_2"]].head())

    # Normalize stars
    scaler = MinMaxScaler()
    data["normalized_stars"] = scaler.fit_transform(data[["stars"]])

    # Recalculate anomaly scores
    data["anomaly_score_1"] = data["sentiment_score_1"] - data["normalized_stars"]
    data["anomaly_score_2"] = data["sentiment_score_2"] - data["normalized_stars"]
    data["anomaly_score"] = (data["anomaly_score_1"] + data["anomaly_score_2"]) / 2

    data.to_csv("dataset/yelp_anomaly.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.histplot(data["anomaly_score"], bins=50, kde=True)
    plt.title("Normalized Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.show()


if __name__ == "__main__":
    main()
