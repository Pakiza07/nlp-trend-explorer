import pandas as pd
import matplotlib.pyplot as plt

def create_visualizations(input_path):
    # Load aggregated data
    df = pd.read_csv(input_path)

    # Convert week to datetime
    df['week_start'] = pd.to_datetime(
        df['week'].str.split('/').str[0]
    )

    # Get unique topics
    topics = df['topic'].unique()

    # Plot topic volume trends
    plt.figure(figsize=(12, 6))

    for topic in topics:
        topic_data = df[df['topic'] == topic]
        plt.plot(
            topic_data['week_start'],
            topic_data['volume'],
            label=topic
        )

    plt.title("Topic Volume Over Time")
    plt.xlabel("Week")
    plt.ylabel("Volume")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save volume plot
    plt.savefig("outputs/topic_volume_trends.png")
    plt.show()

    # Plot sentiment trends
    plt.figure(figsize=(12, 6))

    for topic in topics:
        topic_data = df[df['topic'] == topic]
        plt.plot(
            topic_data['week_start'],
            topic_data['avg_sentiment'],
            label=topic
        )

    plt.title("Sentiment Trends by Topic")
    plt.xlabel("Week")
    plt.ylabel("Average Sentiment")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save sentiment plot
    plt.savefig("outputs/sentiment_trends.png")
    plt.show()


if __name__ == "__main__":
    create_visualizations(
        "data/processed/aggregated_data.csv"
    )