import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    # Compute sentiment score and label
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return score, label

def add_sentiment(input_path, output_path):
    # Load cleaned data
    df = pd.read_csv(input_path)

    # Compute sentiment for each row using original text
    sentiments = df['text'].astype(str).apply(get_sentiment)
    df['sentiment_score'] = sentiments.apply(lambda x: x[0])
    df['sentiment_label'] = sentiments.apply(lambda x: x[1])

    # Save updated dataset
    df.to_csv(output_path, index=False)

    print("Sentiment analysis complete!")
    print(df[['text', 'sentiment_score', 'sentiment_label']].head())

if __name__ == "__main__":
    add_sentiment(
        "data/processed/clean_data.csv",
        "data/processed/clean_data.csv"
    )