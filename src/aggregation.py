import pandas as pd

def aggregate_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Create proper weekly period
    df['week'] = df['date'].dt.to_period('W').astype(str)

    # Debug check for week format
    print("\nDEBUG CHECK:")
    print(df[['date', 'week']].head())

    # Group by topic and week
    grouped = df.groupby(['topic', 'week']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        volume=('text', 'count')
    ).reset_index()

    # Save aggregated data
    grouped.to_csv(output_path, index=False)

    # Preview aggregated output
    print("\nAggregated output:")
    print(grouped.head())


if __name__ == "__main__":
    aggregate_data(
        "data/processed/clean_data.csv",
        "data/processed/aggregated_data.csv"
    )