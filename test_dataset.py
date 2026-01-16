import argparse
import pandas as pd
import numpy as np
import joblib
import os
from data_loader import preprocess_dataset
from features import extract_features_batch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def load_models(models_dir='models'):
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found. Please train models first.")
    
    models = {
        'random_forest': joblib.load(os.path.join(models_dir, 'random_forest.pkl')),
        'logistic_regression': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
        'svm': joblib.load(os.path.join(models_dir, 'svm.pkl'))
    }
    
    scalers = {
        'lr_scaler': joblib.load(os.path.join(models_dir, 'lr_scaler.pkl')),
        'svm_scaler': joblib.load(os.path.join(models_dir, 'svm_scaler.pkl'))
    }
    
    return models, scalers


def predict_batch(urls, model, scaler=None):
    X = extract_features_batch(urls)
    feature_values = X.values
    
    if scaler is not None:
        feature_values = scaler.transform(feature_values)
    
    predictions = model.predict(feature_values)
    
    try:
        probabilities = model.predict_proba(feature_values)
        confidences = probabilities[np.arange(len(predictions)), predictions]
    except:
        confidences = None
    
    return predictions, confidences


def test_dataset(dataset_path, url_column='url', label_column=None, 
                 models_dir='models', output_file=None, model_name='all'):
    print("="*70)
    print("DATASET TESTING")
    print("="*70)
    
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in dataset")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total URLs: {len(df)}")
    
    has_labels = label_column and label_column in df.columns
    if has_labels:
        print(f"Labels found in column: {label_column}")
        print(f"Label distribution:\n{df[label_column].value_counts()}")
    else:
        print("No labels found - will only generate predictions")
    
    print(f"\nLoading models from: {models_dir}")
    models, scalers = load_models(models_dir)
    
    print("\nExtracting features...")
    X = extract_features_batch(df[url_column])
    feature_values = X.values
    print(f"Feature matrix shape: {feature_values.shape}")
    
    results = {}
    model_configs = {
        'random_forest': (models['random_forest'], None, 'Random Forest'),
        'logistic_regression': (models['logistic_regression'], scalers['lr_scaler'], 'Logistic Regression'),
        'svm': (models['svm'], scalers['svm_scaler'], 'SVM')
    }
    
    if model_name == 'all':
        models_to_test = model_configs.keys()
    else:
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_configs.keys())}")
        models_to_test = [model_name]
    
    for model_key in models_to_test:
        model, scaler, model_display_name = model_configs[model_key]
        print(f"\n{'='*70}")
        print(f"Testing with {model_display_name}")
        print(f"{'='*70}")
        
        predictions, confidences = predict_batch(df[url_column], model, scaler)
        
        result_df = df[[url_column]].copy()
        result_df['prediction'] = predictions
        result_df['prediction_label'] = result_df['prediction'].map({0: 'Benign', 1: 'Malicious'})
        
        if confidences is not None:
            result_df['confidence'] = confidences
        
        if has_labels:
            result_df['true_label'] = df[label_column]
            result_df['correct'] = (result_df['prediction'] == result_df['true_label'])
            
            accuracy = accuracy_score(df[label_column], predictions)
            precision = precision_score(df[label_column], predictions, zero_division=0)
            recall = recall_score(df[label_column], predictions, zero_division=0)
            f1 = f1_score(df[label_column], predictions, zero_division=0)
            cm = confusion_matrix(df[label_column], predictions)
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            print(f"\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Benign  Malicious")
            print(f"Actual Benign     {cm[0][0]:4d}      {cm[0][1]:4d}")
            print(f"      Malicious   {cm[1][0]:4d}      {cm[1][1]:4d}")
            
            print(f"\nClassification Report:")
            print(classification_report(df[label_column], predictions, 
                                       target_names=['Benign', 'Malicious']))
        
        prediction_counts = result_df['prediction_label'].value_counts()
        print(f"\nPredictions:")
        for label, count in prediction_counts.items():
            percentage = (count / len(result_df)) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
        
        if confidences is not None:
            print(f"\nConfidence Statistics:")
            print(f"  Mean:   {confidences.mean():.4f}")
            print(f"  Median: {np.median(confidences):.4f}")
            print(f"  Min:    {confidences.min():.4f}")
            print(f"  Max:    {confidences.max():.4f}")
        
        results[model_display_name] = result_df
    
    if output_file:
        if model_name == 'all':
            for model_display_name, result_df in results.items():
                safe_name = model_display_name.lower().replace(' ', '_')
                output_path = output_file.replace('.csv', f'_{safe_name}.csv')
                result_df.to_csv(output_path, index=False)
                print(f"\nResults saved to: {output_path}")
        else:
            result_df = results[list(results.keys())[0]]
            result_df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("TESTING COMPLETE")
    print(f"{'='*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test trained models on a dataset'
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to CSV dataset file'
    )
    parser.add_argument(
        '--url-column',
        type=str,
        default='url',
        help='Name of the URL column (default: url)'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default=None,
        help='Name of the label column (optional, for evaluation)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file to save predictions (optional)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'random_forest', 'logistic_regression', 'svm'],
        help='Which model to use (default: all)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found.")
        return
    
    try:
        test_dataset(
            args.dataset,
            url_column=args.url_column,
            label_column=args.label_column,
            models_dir=args.models_dir,
            output_file=args.output,
            model_name=args.model
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        return


if __name__ == "__main__":
    main()
