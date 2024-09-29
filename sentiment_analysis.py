# sentiment_analysis.py

import os
import json
import logging
import pickle
from typing import List, Tuple

import pandas as pd
import numpy as np

# For model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# For advanced models (e.g., BERT)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Parameters
DATA_PATH = 'data/processed/topic_modeled_conversations.csv'
MODEL_SAVE_PATH = 'models/sentiment_analysis/'
TOKENIZER_NAME = 'bert-base-uncased'
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 3  # e.g., positive, neutral, negative
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

nltk.download('stopwords')
nltk.download('wordnet')

def load_data(path: str) -> pd.DataFrame:
    """
    Load the processed conversation data from a CSV file.
    """
    df = pd.read_csv(path)
    logger.info(f'Data loaded from {path}, shape: {df.shape}')
    return df

def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Preprocess the input text by removing stopwords and lemmatizing.
    """
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the messages and prepare the dataset.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df['cleaned_message'] = df['message'].apply(lambda x: preprocess_text(x, lemmatizer, stop_words))
    logger.info('Text preprocessing completed.')
    return df

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(train_dataset: Dataset, val_dataset: Dataset, model_save_path: str, tokenizer: BertTokenizer):
    """
    Train the BERT model for sentiment analysis.
    """
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    logger.info('BERT model loaded for training.')

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{model_save_path}/logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        return {'accuracy': acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info('Starting training...')
    trainer.train()
    logger.info('Training completed.')

    
