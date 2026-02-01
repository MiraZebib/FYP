import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import Counter

from features import extract_features_batch


def report_class_imbalance(y):
    counts = Counter(y)
    total = len(y)
    
    print("\nClass Distribution:")
    for label, count in sorted(counts.items()):
        pct = (count / total) * 100
        label_name = "Benign" if label == 0 else "Malicious"
        print(f"  {label_name} (Class {label}): {count} ({pct:.2f}%)")
    
    if len(counts) == 2:
        imbalance_ratio = max(counts.values()) / min(counts.values())
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2.0:
            print("WARNING: Significant class imbalance detected")
            print("   Using class_weight='balanced' to handle imbalance")
            return True
    
    return False


def prepare_features_and_labels(df, url_column='url', label_column='label'):
    print("Extracting features from URLs...")
    X = extract_features_batch(df[url_column])
    y = df[label_column].values
    
    print(f"Feature matrix shape: {X.shape}")
    report_class_imbalance(y)
    
    return X.values, y


def train_random_forest(X_train, y_train, n_jobs=-1):
    print("\nTraining Random Forest...")
    
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


def train_logistic_regression(X_train, y_train, scaler=None, handle_imbalance=False):
    print("\nTraining Logistic Regression...")
    
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    class_weight = 'balanced' if handle_imbalance else None
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight)
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='f1', verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler


def train_gradient_boosting(X_train, y_train, handle_imbalance=False):
    print("\nTraining Gradient Boosting...")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    
    class_weight = 'balanced' if handle_imbalance else None
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(
        gb, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_models(df, url_column='url', label_column='label', 
                 test_size=0.2, random_state=42, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    
    X, y = prepare_features_and_labels(df, url_column, label_column)
    
    counts = Counter(y)
    if len(counts) == 2:
        imbalance_ratio = max(counts.values()) / min(counts.values())
        has_imbalance = imbalance_ratio > 2.0
    else:
        has_imbalance = False
    
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    rf_model = train_random_forest(X_train, y_train)
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train, handle_imbalance=has_imbalance)
    gb_model = train_gradient_boosting(X_train, y_train, handle_imbalance=has_imbalance)
    
    print("\nSaving models...")
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression.pkl'))
    joblib.dump(gb_model, os.path.join(models_dir, 'gradient_boosting.pkl'))
    joblib.dump(lr_scaler, os.path.join(models_dir, 'lr_scaler.pkl'))
    
    feature_names = extract_features_batch(['dummy']).columns.tolist()
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    
    print("Models saved successfully!")
    
    return {
        'models': {
            'random_forest': rf_model,
            'logistic_regression': lr_model,
            'gradient_boosting': gb_model
        },
        'scalers': {
            'lr_scaler': lr_scaler
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
