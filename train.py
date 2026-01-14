"""
Training module for malicious URL detection models.

This module trains multiple machine learning classifiers (Random Forest,
Logistic Regression, SVM) and saves them to disk.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

from features import extract_features_batch


def prepare_features_and_labels(df, url_column='url', label_column='label'):
    """
    Extract features from URLs and prepare feature matrix and labels.
    
    Args:
        df (pd.DataFrame): DataFrame with URLs and labels
        url_column (str): Name of the URL column
        label_column (str): Name of the label column
        
    Returns:
        np.ndarray: Feature matrix
        np.ndarray: Label array
    """
    print("Extracting features from URLs...")
    X = extract_features_batch(df[url_column])
    y = df[label_column].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X.values, y


def train_random_forest(X_train, y_train, n_jobs=-1):
    """
    Train a Random Forest classifier with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        n_jobs (int): Number of parallel jobs
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\nTraining Random Forest...")
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1', n_jobs=n_jobs, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_logistic_regression(X_train, y_train, scaler=None):
    """
    Train a Logistic Regression classifier with hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        scaler (StandardScaler): Optional pre-fitted scaler
        
    Returns:
        LogisticRegression: Trained model
        StandardScaler: Fitted scaler
    """
    print("\nTraining Logistic Regression...")
    
    # Scale features for logistic regression
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='f1', verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler


def train_svm(X_train, y_train, scaler=None):
    """
    Train a Support Vector Machine classifier with linear kernel.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        scaler (StandardScaler): Optional pre-fitted scaler
        
    Returns:
        SVC: Trained model
        StandardScaler: Fitted scaler
    """
    print("\nTraining Support Vector Machine (Linear)...")
    
    # Scale features for SVM
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(kernel='linear', random_state=42, probability=True)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler


def train_models(df, url_column='url', label_column='label', 
                 test_size=0.2, random_state=42, models_dir='models'):
    """
    Train all models and save them to disk.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        url_column (str): Name of the URL column
        label_column (str): Name of the label column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        models_dir (str): Directory to save models
        
    Returns:
        dict: Dictionary containing trained models, scalers, and split data
    """
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Prepare features and labels
    X, y = prepare_features_and_labels(df, url_column, label_column)
    
    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    
    # Train Logistic Regression (with scaler)
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
    
    # Train SVM (with scaler, reuse LR scaler for consistency)
    svm_model, svm_scaler = train_svm(X_train, y_train)
    
    # Save models
    print("\nSaving models...")
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    joblib.dump(svm_model, os.path.join(models_dir, 'svm.pkl'))
    joblib.dump(lr_scaler, os.path.join(models_dir, 'lr_scaler.pkl'))
    joblib.dump(svm_scaler, os.path.join(models_dir, 'svm_scaler.pkl'))
    
    # Save feature names for later use
    feature_names = extract_features_batch(['dummy']).columns.tolist()
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    
    print("Models saved successfully!")
    
    return {
        'models': {
            'random_forest': rf_model,
            'logistic_regression': lr_model,
            'svm': svm_model
        },
        'scalers': {
            'lr_scaler': lr_scaler,
            'svm_scaler': svm_scaler
        },
        'data': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        },
        'feature_names': feature_names
    }


if __name__ == "__main__":
    from data_loader import preprocess_dataset
    
    print("Training module - Example usage")
    print("Use train_models() to train all classifiers")
    print("\nExample:")
    print("  df, _ = preprocess_dataset('dataset.csv')")
    print("  results = train_models(df)")
