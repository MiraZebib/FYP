import argparse
import os
from data_loader import preprocess_dataset
from train import train_models
from evaluate import evaluate_all_models


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate malicious URL detection models'
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
        default='label',
        help='Name of the label column (default: label)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found.")
        return
    
    print("="*70)
    print("MALICIOUS URL DETECTION - TRAINING AND EVALUATION")
    print("="*70)
    
    print("\n[Step 1/3] Loading and preprocessing dataset...")
    df, label_mapping = preprocess_dataset(
        args.dataset,
        url_column=args.url_column,
        label_column=args.label_column
    )
    print(f"Label mapping: {label_mapping}")
    
    print("\n[Step 2/3] Training models...")
    results = train_models(
        df,
        url_column=args.url_column,
        label_column=args.label_column,
        test_size=args.test_size,
        models_dir=args.models_dir
    )
    
    print("\n[Step 3/3] Evaluating models...")
    comparison_df = evaluate_all_models(
        results['models'],
        results['scalers'],
        results['data']['X_test'],
        results['data']['y_test'],
        plots_dir=args.plots_dir
    )
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETE")
    print("="*70)
    print(f"\nModels saved to: {args.models_dir}/")
    print(f"Plots saved to: {args.plots_dir}/")
    print("\nTo run the web interface, execute:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
