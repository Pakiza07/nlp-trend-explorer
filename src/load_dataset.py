import kagglehub

# Download latest version
path = kagglehub.dataset_download("crawlfeeds/cnbc-news-headlines-dataset")

print("Path to dataset files:", path)