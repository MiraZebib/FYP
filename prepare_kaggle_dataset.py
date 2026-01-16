import pandas as pd
import argparse
import os


def inspect_dataset(file_path):
    print("="*70)
    print("DATASET INSPECTION")
    print("="*70)
    
    df = pd.read_csv(file_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")
    
    print(f"\nSample values from each column:")
    for col in df.columns:
        sample_values = df[col].dropna().head(3).tolist()
        print(f"  {col}: {sample_values}")
    
    return df


def prepare_dataset(input_file, output_file, url_column=None, label_column=None):
    print("="*70)
    print("PREPARING DATASET")
    print("="*70)
    
    df = pd.read_csv(input_file)
    print(f"\nLoaded dataset: {input_file}")
    print(f"Original shape: {df.shape}")
    
    if url_column is None:
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        url_col = input("\nEnter the name of the URL column: ").strip()
        if url_col not in df.columns:
            raise ValueError(f"Column '{url_col}' not found")
        url_column = url_col
    
    if label_column is None:
        label_col = input("Enter the name of the label column (or press Enter if none): ").strip()
        if label_col and label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found")
        label_column = label_col
    
    if url_column not in df.columns:
        raise ValueError(f"URL column '{url_column}' not found in dataset")
    
    columns_to_keep = [url_column]
    if label_column and label_column in df.columns:
        columns_to_keep.append(label_column)
    
    df_prepared = df[columns_to_keep].copy()
    
    if label_column and label_column in df.columns:
        print(f"\nLabel distribution:")
        print(df_prepared[label_column].value_counts())
    
    df_prepared.to_csv(output_file, index=False)
    print(f"\nPrepared dataset saved to: {output_file}")
    print(f"Final shape: {df_prepared.shape}")
    print(f"Columns: {', '.join(df_prepared.columns)}")
    
    return df_prepared


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and prepare Kaggle datasets for the URL detection system'
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Path to the Kaggle dataset CSV file'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Only inspect the dataset without preparing it'
    )
    parser.add_argument(
        '--url-column',
        type=str,
        default=None,
        help='Name of the URL column (will prompt if not provided)'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default=None,
        help='Name of the label column (optional, will prompt if not provided)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file name (default: prepared_<original_name>)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found.")
        return
    
    try:
        if args.inspect:
            inspect_dataset(args.dataset)
        else:
            if args.output is None:
                base_name = os.path.splitext(os.path.basename(args.dataset))[0]
                args.output = f"prepared_{base_name}.csv"
            
            prepare_dataset(
                args.dataset,
                args.output,
                url_column=args.url_column,
                label_column=args.label_column
            )
            
            print("\n" + "="*70)
            print("NEXT STEPS:")
            print("="*70)
            print(f"\n1. Train models:")
            print(f"   python main.py {args.output} --url-column {args.url_column}", end="")
            if args.label_column:
                print(f" --label-column {args.label_column}")
            else:
                print()
            
            print(f"\n2. Or test on this dataset (if you have trained models):")
            print(f"   python test_dataset.py {args.output} --url-column {args.url_column}", end="")
            if args.label_column:
                print(f" --label-column {args.label_column}")
            else:
                print()
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        return


if __name__ == "__main__":
    main()
