import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(query, top_k=5):

    # Get project root directory 
    BASE_DIR = os.path.dirname(os.getcwd())

    # Load processed dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/clean_data.csv"))

    # Load precomputed embeddings
    embeddings = np.load(os.path.join(BASE_DIR, "data/processed/embeddings.npy"))

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert query into embedding vector
    query_embedding = model.encode([query])

    # Compute similarity between query and dataset embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get indices of top matching results
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Print search results
    print(f"\nTop results for: '{query}'\n")

    # Display top matching rows
    for idx in top_indices:
        print(df.iloc[idx]["text"])
        print(df.iloc[idx]["topic"])
        print(df.iloc[idx]["sentiment_label"])
        print(f"Score: {similarities[idx]:.4f}")
        print("-" * 80)


if __name__ == "__main__":
    semantic_search("stock market crash")