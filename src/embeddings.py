import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings from clean text
    embeddings = model.encode(
        df['clean_text'].tolist(),
        show_progress_bar=True
    )

    # Save embeddings as numpy array
    np.save(output_path, embeddings)

    print("Embeddings generated")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    generate_embeddings(
        "data/processed/clean_data.csv",
        "data/processed/embeddings.npy"
    )