import pandas as pd

def add_date_features(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Ensure date column exists
    if 'date' not in df.columns:
        raise ValueError("Column 'date' not found in dataset")

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Extract features
    df['day'] = df['date'].dt.date
    df['week'] = df['date'].dt.isocalendar().week

    # Save updated dataset
    df.to_csv(output_path, index=False)

    print("Date features added!")
    print(df[['date', 'day', 'week']].head())


if __name__ == "__main__":
    add_date_features(
        "data/processed/clean_data.csv",
        "data/processed/clean_data.csv"
    )