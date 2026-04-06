import pandas as pd
import spacy

# Load spaCy English model (disable parser & NER for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    # Lowercase, tokenize, remove stopwords/punctuation, lemmatize
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(tokens)

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)


    # Clean text
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    # Save processed CSV
    df.to_csv(output_path, index=False)

    # Show sample
    print("Preprocessing complete!")
    print(df[['text', 'clean_text']].head())

if __name__ == "__main__":
    preprocess_data(
        "data/raw/base_data.csv",
        "data/processed/clean_data.csv"
    )