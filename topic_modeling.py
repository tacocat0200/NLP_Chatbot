# topic_modeling.py

import os
import json
import logging
import pickle
import pandas as pd
from typing import List, Tuple

# For LDA
import gensim
from gensim import corpora
from gensim.models import LdaModel

# For BERT-based Topic Modeling
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# For preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Configuration Parameters
DATA_PATH = 'data/processed/cleaned_conversations.json'
LDA_MODEL_PATH = 'models/topic_modeling/lda_model.pkl'
BERT_MODEL_PATH = 'models/topic_modeling/bert_topic_model.pkl'
NUM_TOPICS = 5  # Top 5 trending issues
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'  # Lightweight BERT model
KMEANS_CLUSTER = 5  # Number of clusters for KMeans

# Data Preprocessing Functions

def load_data(path: str) -> List[str]:
    """
    Load cleaned conversation messages from a JSON file.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    # Assuming each entry has a 'message' field
    messages = [entry['message'] for entry in data if entry['speaker'] == 'user']
    return messages

def preprocess_text(texts: List[str]) -> List[List[str]]:
    """
    Preprocess the text data:
    - Lowercasing
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_texts = []
    for text in texts:
        tokens = gensim.utils.simple_preprocess(text, deacc=True)  # Removes punctuation
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        processed_texts.append(tokens)
    return processed_texts

# LDA Topic Modeling Functions

def train_lda_model(processed_texts: List[List[str]], num_topics: int = NUM_TOPICS) -> Tuple[LdaModel, corpora.Dictionary]:
    """
    Train an LDA model using Gensim.
    """
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    return lda_model, dictionary

def get_lda_topics(lda_model: LdaModel, dictionary: corpora.Dictionary, num_words: int = 10) -> List[str]:
    """
    Extract topics from the LDA model.
    """
    topics = []
    for idx, topic in lda_model.print_topics(num_topics=NUM_TOPICS, num_words=num_words):
        topic_keywords = ", ".join([word.split("*")[1].strip().strip('"') for word in topic.split(" + ")])
        topics.append(f"Topic {idx + 1}: {topic_keywords}")
    return topics

# BERT-based Topic Modeling Functions

def generate_bert_embeddings(texts: List[str], model_name: str = BERT_MODEL_NAME) -> np.ndarray:
    """
    Generate BERT embeddings for the given texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def train_bert_topic_model(embeddings: np.ndarray, num_clusters: int = KMEANS_CLUSTER) -> KMeans:
    """
    Train a KMeans clustering model on BERT embeddings.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans

def get_bert_topics(kmeans_model: KMeans, texts: List[str]) -> List[str]:
    """
    Assign cluster labels as topics to the texts.
    """
    clusters = kmeans_model.labels_
    # For interpretability, you might want to extract top terms per cluster
    # Here, we'll just return the cluster numbers as topic labels
    topics = [f"Topic {cluster + 1}" for cluster in clusters]
    return topics

# Model Evaluation Functions

def evaluate_lda_model(lda_model: LdaModel, corpus: List[List[int]], dictionary: corpora.Dictionary) -> None:
    """
    Print coherence score for the LDA model.
    """
    from gensim.models import CoherenceModel
    coherence_model_lda = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logging.info(f'LDA Model Coherence Score: {coherence_lda}')

def evaluate_kmeans_model(kmeans_model: KMeans, embeddings: np.ndarray) -> None:
    """
    Print silhouette score for the KMeans model.
    """
    from sklearn.metrics import silhouette_score
    score = silhouette_score(embeddings, kmeans_model.labels_)
    logging.info(f'KMeans Silhouette Score: {score}')

# Utility Functions

def save_lda_model(lda_model: LdaModel, dictionary: corpora.Dictionary, path: str = LDA_MODEL_PATH) -> None:
    """
    Save the LDA model and dictionary to disk.
    """
    lda_model.save(path)
    dictionary.save(f"{path}_dictionary.dict")
    logging.info(f'LDA model and dictionary saved to {path}')

def load_lda_model(path: str = LDA_MODEL_PATH) -> Tuple[LdaModel, corpora.Dictionary]:
    """
    Load the LDA model and dictionary from disk.
    """
    lda_model = LdaModel.load(path)
    dictionary = corpora.Dictionary.load(f"{path}_dictionary.dict")
    logging.info(f'LDA model and dictionary loaded from {path}')
    return lda_model, dictionary

def save_bert_model(kmeans_model: KMeans, path: str = BERT_MODEL_PATH) -> None:
    """
    Save the KMeans model to disk.
    """
    with open(path, 'wb') as f:
        pickle.dump(kmeans_model, f)
    logging.info(f'KMeans model saved to {path}')

def load_bert_model(path: str = BERT_MODEL_PATH) -> KMeans:
    """
    Load the KMeans model from disk.
    """
    with open(path, 'rb') as f:
        kmeans_model = pickle.load(f)
    logging.info(f'KMeans model loaded from {path}')
    return kmeans_model

# Main Execution Flow

def main():
    # Load and preprocess data
    logging.info('Loading data...')
    raw_messages = load_data(DATA_PATH)
    logging.info(f'Loaded {len(raw_messages)} messages.')
    
    logging.info('Preprocessing data...')
    processed_texts = preprocess_text(raw_messages)
    
    # LDA Topic Modeling
    logging.info('Training LDA model...')
    lda_model, dictionary = train_lda_model(processed_texts)
    
    logging.info('Extracting LDA topics...')
    lda_topics = get_lda_topics(lda_model, dictionary)
    for topic in lda_topics:
        logging.info(topic)
    
    # Save LDA model
    save_lda_model(lda_model, dictionary)
    
    # BERT-based Topic Modeling
    logging.info('Generating BERT embeddings...')
    bert_embeddings = generate_bert_embeddings(raw_messages)
    
    logging.info('Training BERT-based KMeans model...')
    kmeans_model = train_bert_topic_model(bert_embeddings)
    
    # Assign topics
    bert_topics = get_bert_topics(kmeans_model, raw_messages)
    
    # Save BERT-based model
    save_bert_model(kmeans_model)
    
    # Evaluate Models
    logging.info('Evaluating LDA model...')
    lda_corpus = [dictionary.doc2bow(text) for text in processed_texts]
    evaluate_lda_model(lda_model, lda_corpus, dictionary)
    
    logging.info('Evaluating KMeans model...')
    evaluate_kmeans_model(kmeans_model, bert_embeddings)
    
    # Optionally, save the topics alongside the original data
    df = pd.read_json(DATA_PATH)
    df['lda_topic'] = [max(lda_model.get_document_topics(dictionary.doc2bow(text), minimum_probability=0), key=lambda x: x[1])[0] + 1 for text in processed_texts]
    df['bert_topic'] = bert_topics
    df.to_csv('data/processed/topic_modeled_conversations.csv', index=False)
    logging.info('Saved topic-modeled conversations to CSV.')

if __name__ == '__main__':
    main()
