"""
Data loading and preprocessing module for malicious URL detection.

This module handles loading CSV datasets, cleaning URLs, removing duplicates,
and encoding labels for machine learning.
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re


def load_dataset(file_path, url_column='url', label_column='label'):
    """
    Load a CSV dataset containing URLs and labels.
    
    Args:
        file_path (str): Path to the CSV file
        url_column (str): Name of the column containing URLs
        label_column (str): Name of the column containing labels
        
    Returns:
        pd.DataFrame: DataFrame with URLs and labels
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
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
    """
    Clean and normalize a URL string.
    
    Args:
        url (str): Raw URL string
        
    Returns:
        str: Cleaned URL string
    """
    if pd.isna(url) or url is None:
        return ""
    
    url = str(url).strip()
    
    # Remove leading/trailing whitespace
    url = url.strip()
    
    # Add protocol if missing (for parsing purposes)
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    return url


def remove_duplicates(df, url_column='url'):
    """
    Remove duplicate URLs from the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        url_column (str): Name of the URL column
        
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset=[url_column], keep='first')
    removed_count = initial_count - len(df_cleaned)
    
    if removed_count > 0:
        print(f"Removed {removed_count} duplicate URLs")
    
    return df_cleaned


def encode_labels(df, label_column='label'):
    """
    Encode string labels to numeric values.
    Maps: benign/0 -> 0, malicious/1 -> 1
    
    Args:
        df (pd.DataFrame): Input DataFrame with labels
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: DataFrame with encoded labels
        dict: Label mapping dictionary
    """
    df = df.copy()
    
    # Normalize label values
    df[label_column] = df[label_column].astype(str).str.lower().str.strip()
    
    # Create mapping
    unique_labels = df[label_column].unique()
    label_mapping = {}
    
    # Map benign/0 -> 0, malicious/1 -> 1
    for label in unique_labels:
        if 'benign' in label or label == '0' or label == '0.0':
            label_mapping[label] = 0
        elif 'malicious' in label or 'malware' in label or label == '1' or label == '1.0':
            label_mapping[label] = 1
        else:
            # Default: assume 0/1 encoding
            try:
                label_mapping[label] = int(float(label))
            except:
                label_mapping[label] = 0
    
    # Apply mapping
    df[label_column] = df[label_column].map(label_mapping)
    
    # Handle any NaN values (shouldn't happen, but just in case)
    df[label_column] = df[label_column].fillna(0)
    
    return df, label_mapping


def preprocess_dataset(file_path, url_column='url', label_column='label'):
    """
    Complete preprocessing pipeline: load, clean, remove duplicates, encode labels.
    
    Args:
        file_path (str): Path to the CSV file
        url_column (str): Name of the URL column
        label_column (str): Name of the label column
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame
        dict: Label mapping dictionary
    """
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
    # Example usage
    print("Data loader module - Example usage")
    print("Use preprocess_dataset() to load and preprocess your CSV file")
