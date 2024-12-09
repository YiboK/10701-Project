import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import swifter  # For parallel processing

# VADER
analyzer = SentimentIntensityAnalyzer()
feature_names= ["scores_pooling_mean", "scores_pooling_max", "scores_pooling_min", "scores_pooling_std", "scores_pooling_sum",
            "scores_concat",
            "scores_shift_psm", "scores_shift_ratio"]

# Get sections' compount sentiment scores
def get_compound_scores(text):
    sections = re.split(r'[.,!?;]', text) # Split the text into sections using punctuation
    sections = [section.strip() for section in sections if section.strip()]

    compound_scores = [analyzer.polarity_scores(section)['compound'] for section in sections if section] # compound score, ignoring blanks
    return compound_scores

# Pool scores: mean, max, min, std, sum
def generate_pooling_features(scores):
    if len(scores) == 0:
        return [0, 0, 0, 0, 0]
    return [
        np.mean(scores),   # Mean
        np.max(scores),    # Max
        np.min(scores),    # Min
        np.std(scores),    # Standard Deviation
        np.sum(scores)     # Sum
    ]

# Concat scores with -
def generate_concat_feature(scores):
  
  # print("concat:", scores)
  joined_scores = "|".join(map(str, scores))
  return [joined_scores]

# Calcualte consecutive pair-wise shifts: PSM(squared difference), sign change ratio
def generate_shift_features(scores):
  polarity_changes = [
    1 if np.sign(scores[i]) != np.sign(scores[i-1]) else 0
    for i in range(1, len(scores))  # Loop through consecutive pairs
  ]

  polarity_shifts = [
        (scores[i] - scores[i+1])**2  # Magnitude of change: Squared difference
        for i in range(len(polarity_changes))
        if polarity_changes[i] == 1 # Only count sign changes
    ]
  psm = sum(polarity_shifts)  # Combine into a single number
  change_ratio = sum(polarity_changes) / len(polarity_changes) if len(polarity_changes) != 0 else 0
  return [psm, change_ratio]

# Features: pooling, concat, pair-wise distance
def process_line(text):
    compound_scores = get_compound_scores(text)

    # pooling
    pooling_features = generate_pooling_features(compound_scores)

    # concat
    concat_features = generate_concat_feature(compound_scores)

    # pair-wise distance
    shift_features = generate_shift_features(compound_scores)

    features = pd.Series(pooling_features + concat_features + shift_features, index=feature_names)
    return features

def main():
    df = pd.read_csv("dataset/yelp_appended.csv")

    # Step 5: Apply Sentiment Analysis and Pooling Using Swifter for Speed
    df[feature_names] = df['text'].swifter.apply(process_line)

    output_file = "dataset/yelp_pseudo_features.csv"
    df.to_csv(output_file, index=False)

    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    main()