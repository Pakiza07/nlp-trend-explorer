import pandas as pd

def detect_patterns(input_path):
    # Load aggregated dataset
    df = pd.read_csv(input_path)

    # Convert week start to datetime
    df['week_start'] = pd.to_datetime(
        df['week'].str.split('/').str[0]
    )

    # Sort values for proper trend analysis
    df = df.sort_values(by=['topic', 'week_start'])

    # Compute week-to-week volume change
    df['volume_change'] = df.groupby('topic')['volume'].diff()

    # Compute week-to-week sentiment change
    df['sentiment_change'] = df.groupby('topic')['avg_sentiment'].diff()

    # Detect large spikes in volume
    volume_spikes = df[df['volume_change'] > df['volume_change'].std()]

    # Detect large sentiment shifts
    sentiment_spikes = df[
        abs(df['sentiment_change']) > df['sentiment_change'].std()
    ]

    # Display findings
    print("\nVolume spikes:")
    print(volume_spikes[
        ['topic', 'week', 'volume', 'volume_change']
    ].head(10))

    print("\nSentiment shifts:")
    print(sentiment_spikes[
        ['topic', 'week', 'avg_sentiment', 'sentiment_change']
    ].head(10))


if __name__ == "__main__":
    detect_patterns(
        "data/processed/aggregated_data.csv"
    )