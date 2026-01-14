import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re


def load_dataset(file_path, url_column='url', label_column='label'):
    try:
        df = pd.read_csv(file_path)
        
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found in dataset")
            
        return df[[url_column, label_column]]
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def clean_url(url):
    if pd.isna(url) or url is None:
        return ""
    
    url = str(url).strip()
    
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    return url


def remove_duplicates(df, url_column='url'):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=[url_column], keep='first')
    removed_count = initial_count - len(df_cleaned)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate URLs")
    
    return df_cleaned


def encode_labels(df, label_column='label'):
    df = df.copy()
    
    df[label_column] = df[label_column].astype(str).str.lower().str.strip()
    
    unique_labels = df[label_column].unique()
    label_mapping = {}
    
    for label in unique_labels:
        if 'benign' in label or label == '0' or label == '0.0':
            label_mapping[label] = 0
        elif 'malicious' in label or 'malware' in label or label == '1' or label == '1.0':
            label_mapping[label] = 1
        else:
            try:
                label_mapping[label] = int(float(label))
            except:
                label_mapping[label] = 0
    
    df[label_column] = df[label_column].map(label_mapping)
    df[label_column] = df[label_column].fillna(0)
    
    return df, label_mapping


def preprocess_dataset(file_path, url_column='url', label_column='label'):
    print("Loading dataset...")
    df = load_dataset(file_path, url_column, label_column)
    
    print("Cleaning URLs...")
    df[url_column] = df[url_column].apply(clean_url)
    
    print("Removing duplicates...")
    df = remove_duplicates(df, url_column)
    
    print("Encoding labels...")
    df, label_mapping = encode_labels(df, label_column)
    
    print(f"Preprocessing complete. Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df[label_column].value_counts()}")
    
    return df, label_mapping


if __name__ == "__main__":
    print("Data loader module - Example usage")
    print("Use preprocess_dataset() to load and preprocess your CSV file")
