# Malicious URL Detection System

A Python-based machine learning system that detects malicious URLs using lexical and structural feature analysis. The system analyzes URLs as plain text only and **never opens, visits, or executes** the URLs.

## Features

- **Lexical Feature Extraction**: Analyzes URL structure, character patterns, entropy, and more
- **Multiple ML Models**: Random Forest, Logistic Regression, and Support Vector Machine
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- **Web Interface**: User-friendly Streamlit application for real-time URL classification
- **Offline Analysis**: All processing is performed locally without network requests

## Project Structure

```
.
├── data_loader.py      # Dataset loading and preprocessing
├── features.py         # Lexical feature extraction
├── train.py           # Model training and saving
├── evaluate.py        # Model evaluation and visualization
├── test_dataset.py    # Test models on new datasets
├── app.py             # Streamlit web interface
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Format

The system expects a CSV file with at least one column:
- **URL column**: Contains the URL strings (required)
- **Label column**: Contains labels (benign/malicious or 0/1) - optional for testing

Example CSV format:
```csv
url,label
https://example.com,benign
http://suspicious-site.com/login,malicious
```

### Using Kaggle Datasets

#### Option 1: Download Directly from Kaggle (Recommended)

Download and prepare a dataset directly from Kaggle using the Kaggle Hub API:

```bash
# Download and prepare in one step
python download_kaggle_dataset.py "himadri07/malicious-urls-dataset-15k-rows"
```

The script will:
- Download the dataset from Kaggle
- Show you the CSV files found
- Let you select which file to use
- Inspect the dataset structure
- Help you select URL and label columns
- Create a prepared dataset ready for training

**With specific column names:**
```bash
python download_kaggle_dataset.py "himadri07/malicious-urls-dataset-15k-rows" --url-column url --label-column label --output my_dataset.csv
```

**Note:** You'll need to authenticate with Kaggle first. Set up your Kaggle API credentials:
1. Go to https://www.kaggle.com/settings
2. Click "Create New Token" to download `kaggle.json`
3. Place it in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows)

#### Option 2: Manual Download

If you manually downloaded a dataset from Kaggle:

1. **Inspect the dataset** to see its structure:
   ```bash
   python prepare_kaggle_dataset.py your_kaggle_dataset.csv --inspect
   ```

2. **Prepare the dataset** (extract URL and label columns):
   ```bash
   python prepare_kaggle_dataset.py your_kaggle_dataset.csv --url-column url --label-column label --output prepared_dataset.csv
   ```

   Or run interactively (it will prompt you for column names):
   ```bash
   python prepare_kaggle_dataset.py your_kaggle_dataset.csv --output prepared_dataset.csv
   ```

3. **Train models** with the prepared dataset:
   ```bash
   python main.py prepared_dataset.csv --url-column url --label-column label
   ```

## Usage

### 1. Prepare Your Dataset

Place your CSV dataset in the project directory. The dataset should contain URLs and labels.

### 2. Train Models

Create a training script or run the training module:

```python
from data_loader import preprocess_dataset
from train import train_models

# Load and preprocess dataset
df, label_mapping = preprocess_dataset(
    'your_dataset.csv',
    url_column='url',      # Name of URL column
    label_column='label'   # Name of label column
)

# Train all models
results = train_models(df)
```

Or use command line:
```python
python -c "from data_loader import preprocess_dataset; from train import train_models; df, _ = preprocess_dataset('dataset.csv'); train_models(df)"
```

This will:
- Extract features from URLs
- Train Random Forest, Logistic Regression, and SVM models
- Save models to the `models/` directory

### 3. Evaluate Models

```python
from evaluate import evaluate_all_models

# Evaluate all models
comparison_df = evaluate_all_models(
    results['models'],
    results['scalers'],
    results['data']['X_test'],
    results['data']['y_test']
)
```

This generates:
- Confusion matrices for each model
- ROC curve comparison
- Performance comparison table (saved as CSV)
- Detailed classification reports

### 4. Run Web Interface

```bash
streamlit run app.py
```

The web interface will open in your browser, allowing you to:
- Input URLs for classification
- View predictions with confidence scores
- See feature importance explanations
- Compare different models

### 5. Test Models on New Datasets

Test your trained models on new datasets:

```bash
# Test with all models (requires labels for evaluation)
python test_dataset.py your_dataset.csv --label-column label

# Test with a specific model
python test_dataset.py your_dataset.csv --model random_forest --label-column label

# Test without labels (prediction only)
python test_dataset.py your_dataset.csv

# Save predictions to CSV
python test_dataset.py your_dataset.csv --output predictions.csv --label-column label
```

The script will:
- Load trained models
- Extract features from URLs in your dataset
- Generate predictions for all URLs
- Calculate performance metrics (if labels provided)
- Display confusion matrix and classification report
- Optionally save results to CSV

## Extracted Features

The system extracts the following lexical and structural features:

- **Length Features**: URL length, path length, query length, hostname length
- **Character Counts**: Digits, letters, special characters
- **Symbol Counts**: Dots, dashes, slashes, @, ?, =, &, %
- **Structural Features**: 
  - Number of subdomains
  - Presence of IP address
  - HTTPS usage
  - URL entropy
  - Suspicious token detection
- **Ratios**: Digit ratio, special character ratio

## Model Details

### Random Forest
- Ensemble method with multiple decision trees
- Provides feature importance for interpretability
- Hyperparameter tuning: n_estimators, max_depth, min_samples_split, min_samples_leaf

### Logistic Regression
- Linear classifier with L1/L2 regularization
- Features are standardized before training
- Hyperparameter tuning: C, penalty, solver

### Support Vector Machine
- Linear kernel SVM
- Features are standardized before training
- Hyperparameter tuning: C, gamma

## Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

## Important Notes

⚠️ **Ethics and Constraints**:
- This system analyzes URLs as plain text only
- URLs are **never opened, visited, or executed**
- No network requests are made
- No user data is collected or stored
- All analysis is performed offline

⚠️ **Dataset Requirements**:
- Use only publicly available, anonymized datasets
- Ensure compliance with data usage policies
- The system expects balanced or reasonably balanced datasets

## Troubleshooting

### Models Not Found
If you see "Models directory not found", ensure you've trained the models first using `train.py`.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Memory Issues
For large datasets, consider:
- Sampling the dataset before training
- Reducing the number of trees in Random Forest
- Using a smaller test set size

## Example Workflow

### Quick Start (Command Line)

```bash
# 1. Train and evaluate models
python main.py your_dataset.csv

# 2. Test models on a new dataset
python test_dataset.py your_test_dataset.csv --label-column label

# 3. Run web interface
streamlit run app.py
```

### Programmatic Usage

```python
# 1. Load and preprocess
from data_loader import preprocess_dataset
df, _ = preprocess_dataset('malicious_urls.csv')

# 2. Train models
from train import train_models
results = train_models(df, test_size=0.2)

# 3. Evaluate
from evaluate import evaluate_all_models
comparison = evaluate_all_models(
    results['models'],
    results['scalers'],
    results['data']['X_test'],
    results['data']['y_test']
)

# 4. Use in application
# Run: streamlit run app.py
```

## License

This project is intended for academic evaluation, demonstration, and further extension.

## Contributing

This is an academic project. For extensions or improvements:
1. Ensure all analysis remains offline
2. Maintain ethical data usage practices
3. Document any new features or changes

## Contact

For questions or issues related to this implementation, please refer to the project documentation or academic supervisor.
