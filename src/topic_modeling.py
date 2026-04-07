import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_topic_modeling(input_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure clean_text exists
    if 'clean_text' not in df.columns:
        raise ValueError("Column 'clean_text' not found in dataset")

    # Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=5)
    X = vectorizer.fit_transform(df['clean_text'])

    # LDA model
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    # Get feature names (words)
    words = vectorizer.get_feature_names_out()

    # Print top words per topic
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-10:]]
        print(f"\nTopic {i+1}: {', '.join(top_words)}")

if __name__ == "__main__":
    run_topic_modeling("data/processed/clean_data.csv")