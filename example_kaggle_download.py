import kagglehub
import os

dataset_name = "himadri07/malicious-urls-dataset-15k-rows"

print("Downloading dataset from Kaggle...")
print(f"Dataset: {dataset_name}")

path = kagglehub.dataset_download(dataset_name)

print(f"\nDataset downloaded to: {path}")

csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

print(f"\nFound {len(csv_files)} CSV file(s):")
for csv_file in csv_files:
    file_size = os.path.getsize(csv_file) / (1024 * 1024)
    print(f"  - {os.path.basename(csv_file)} ({file_size:.2f} MB)")
    print(f"    Full path: {csv_file}")

if csv_files:
    print(f"\nTo prepare this dataset, run:")
    print(f"  python prepare_kaggle_dataset.py \"{csv_files[0]}\" --output prepared_dataset.csv")
    
    print(f"\nOr use the automated download script:")
    print(f"  python download_kaggle_dataset.py \"{dataset_name}\"")
