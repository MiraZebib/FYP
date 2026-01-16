import kagglehub
import os
import pandas as pd
import argparse
from prepare_kaggle_dataset import inspect_dataset, prepare_dataset


def download_and_prepare(dataset_name, url_column=None, label_column=None, output_file=None):
    print("="*70)
    print("DOWNLOADING FROM KAGGLE")
    print("="*70)
    
    print(f"\nDownloading dataset: {dataset_name}")
    print("This may take a few moments...")
    
    try:
        path = kagglehub.dataset_download(dataset_name)
        print(f"\nDataset downloaded to: {path}")
        
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        print(f"\nFound {len(csv_files)} CSV file(s):")
        for i, csv_file in enumerate(csv_files, 1):
            file_size = os.path.getsize(csv_file) / (1024 * 1024)
            print(f"  {i}. {os.path.basename(csv_file)} ({file_size:.2f} MB)")
        
        if len(csv_files) == 1:
            dataset_file = csv_files[0]
            print(f"\nUsing: {os.path.basename(dataset_file)}")
        else:
            print("\nMultiple CSV files found. Please select one:")
            for i, csv_file in enumerate(csv_files, 1):
                print(f"  {i}. {os.path.basename(csv_file)}")
            
            choice = int(input("\nEnter file number: ")) - 1
            if choice < 0 or choice >= len(csv_files):
                raise ValueError("Invalid selection")
            dataset_file = csv_files[choice]
        
        print(f"\nInspecting dataset...")
        inspect_dataset(dataset_file)
        
        if output_file is None:
            base_name = dataset_name.split('/')[-1]
            output_file = f"prepared_{base_name}.csv"
        
        print(f"\nPreparing dataset...")
        if url_column is None or label_column is None:
            print("\nYou'll be prompted to select columns interactively.")
        
        prepare_dataset(
            dataset_file,
            output_file,
            url_column=url_column,
            label_column=label_column
        )
        
        print("\n" + "="*70)
        print("DOWNLOAD AND PREPARATION COMPLETE")
        print("="*70)
        print(f"\nPrepared dataset saved to: {output_file}")
        print(f"\nNext steps:")
        print(f"  1. Train models:")
        print(f"     python main.py {output_file} --url-column {url_column or 'url'} --label-column {label_column or 'label'}")
        print(f"\n  2. Or test on this dataset (if models already trained):")
        print(f"     python test_dataset.py {output_file} --url-column {url_column or 'url'} --label-column {label_column or 'label'}")
        
        return output_file
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Download dataset from Kaggle and prepare it for the URL detection system'
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Kaggle dataset name (e.g., "username/dataset-name")'
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
        help='Output file name (default: prepared_<dataset_name>.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        download_and_prepare(
            args.dataset,
            url_column=args.url_column,
            label_column=args.label_column,
            output_file=args.output
        )
    except Exception as e:
        print(f"\nFailed to download and prepare dataset: {str(e)}")
        return


if __name__ == "__main__":
    main()
