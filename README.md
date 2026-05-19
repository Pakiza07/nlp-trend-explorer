# NLP Trend Explorer

An NLP-based analytics system for exploring topic trends, sentiment patterns, and semantic relationships in financial news data.

This project combines preprocessing, sentiment analysis, topic modeling, trend analysis, and semantic search into a unified NLP analytics pipeline.

---

# Features

- Text preprocessing using spaCy
- Sentiment analysis using VADER
- Topic modeling using LDA
- Topic assignment and labeling
- Temporal trend analysis
- Trend spike and sentiment shift detection
- Semantic search using sentence embeddings
- Interactive notebook walkthrough

---

# Tech Stack

- Python
- pandas
- NumPy
- spaCy
- NLTK
- scikit-learn
- sentence-transformers
- matplotlib
- Jupyter Notebook

---

# Project Structure

```text
nlp-trend-explorer/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_setup.ipynb
│   └── trend_explorer.ipynb
|
├── outputs/
│   ├── topic_volume_trends.png
│   ├── sentiment_trends.png
│   └── screenshots/
│
├── src/
│   ├── aggregation.py
|   ├── data_features.py
|   ├── load_dataset.py
|   ├── pattern_detection.py
|   ├── preprocessing.py
│   ├── sentiment.py
│   ├── topic_modeling.py
│   ├── topic_labels.py
│   ├── topic_assignment.py
│   ├── visualization.py
│   ├── embeddings.py
│   └── semantic_search.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses financial news and text data for analyzing:

- topic distributions
- sentiment behavior
- temporal trends
- semantic similarity

The dataset is processed through multiple NLP stages before analysis.

---

# NLP Pipeline

## 1. Text Preprocessing

The preprocessing stage includes:

- lowercasing
- punctuation removal
- stopword removal
- lemmatization using spaCy

---

## 2. Sentiment Analysis

Sentiment analysis is performed using VADER from NLTK.

Each text receives:

- sentiment score
- sentiment label

Possible labels:
- Positive
- Negative
- Neutral

---

## 3. Topic Modeling

Topic modeling is implemented using Latent Dirichlet Allocation (LDA).

Generated topics were manually interpreted and labeled into categories such as:

- Stock Market
- Company News
- Financial Personalities
- News Metadata

---

## 4. Trend Analysis

Temporal aggregation was used to analyze:

- topic frequency over time
- average sentiment trends
- volatility patterns

---

## 5. Pattern Detection

The project identifies:

- topic volume spikes
- sentiment polarity shifts
- temporal fluctuations in topic distribution

---

## 6. Semantic Search

Sentence embeddings were generated using Sentence Transformers.

Users can search semantically related financial texts using natural language queries.

Example:

```python
semantic_search("stock market crash")
```

---

# Example Outputs

## Topic Distribution

<img width="942" height="710" alt="Screenshot 2026-05-19 182310" src="https://github.com/user-attachments/assets/3eef2289-caf9-44a7-bc14-20eec2eb3f2e" />


---

## Topic Volume Trends

<img width="1240" height="617" alt="Screenshot 2026-05-19 181613" src="https://github.com/user-attachments/assets/e0bbc77b-8dd5-430d-99d0-c464edbb465f" />

---

## Sentiment Trends

<img width="1248" height="626" alt="Screenshot 2026-05-19 181656" src="https://github.com/user-attachments/assets/f21ea847-2a7c-41c0-a78b-1160a3e02bce" />

---

## Semantic Search Demo

<img width="1780" height="752" alt="Screenshot 2026-05-19 182655" src="https://github.com/user-attachments/assets/d371f057-4f5d-4d71-963f-d0dcec1c4b4b" />

---

# Key Insights

- Stock Market was the dominant topic across the dataset.
- Company News exhibited the highest volatility in sentiment and volume.
- Trend spikes appeared during specific time windows.
- Semantic search successfully retrieved contextually related financial texts.

---

# Installation

Clone the repository:

```bash
git clone <https://github.com/Pakiza07/nlp-trend-explorer>
```

Move into project directory:

```bash
cd nlp-trend-explorer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run preprocessing:

```bash
python src/preprocessing.py
```

Run sentiment analysis:

```bash
python src/sentiment.py
```

Run topic modeling:

```bash
python src/topic_modeling.py
```

Run semantic search:

```bash
python src/semantic_search.py
```

Or explore the interactive notebook:

```text
notebooks/trend_explorer.ipynb
```

---

# Future Improvements

Possible future extensions include:

- real-time news ingestion
- interactive dashboard
- transformer fine-tuning
- live trend monitoring
- advanced visualization system

---

# Author

Created as an end-to-end NLP analytics and trend exploration project using Python and modern NLP techniques.
