import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from topic_labels import topic_labels  # import labels

def assign_topics(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure clean_text exists
    if 'clean_text' not in df.columns:
        raise ValueError("Column 'clean_text' not found")

    # Vectorize text using TF-IDF
    vectorizer = CountVectorizer(max_df=0.95, min_df=5)
    X = vectorizer.fit_transform(df['clean_text'])

    # Train LDA model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # Get topic distribution per row
    topic_dist = lda.transform(X)

    # Assign dominant topic id
    df['topic_id'] = topic_dist.argmax(axis=1)

    # Map topic ids to labels
    df['topic'] = df['topic_id'].map(topic_labels)

    # Print topic distribution
    print("\nTopic distribution:")
    print(df['topic'].value_counts())

    # Save updated dataset
    df.to_csv(output_path, index=False)

    # Preview results
    print("\nSample output:")
    print(df[['clean_text', 'topic']].head())


if __name__ == "__main__":
    assign_topics(
        "data/processed/clean_data.csv",
        "data/processed/clean_data.csv"
    )