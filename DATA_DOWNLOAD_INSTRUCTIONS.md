# Data Download Instructions

Due to the large size of the dataset files, they are not included in this GitHub repository. Please follow these instructions to download and set up the data.

## Dataset Information

This project uses the **UNSW-NB15** dataset for network intrusion detection. The dataset contains various network traffic features for anomaly detection.

## Files Not Included in Repository

The following large files are excluded from Git:

1. **Raw Data Files** (in `data/raw/`):
   - `UNSW_NB15_training-set.csv` (30.8 MB)
   - `UNSW_NB15_testing-set.csv` (14.7 MB)

2. **Processed Data Files** (in `data/processed/`):
   - `X_train.csv` (473 MB)
   - `X_test.csv` (101 MB)
   - `X_val.csv` (101 MB)

## Download Options

### Option 1: Download from Original Source
1. Visit the UNSW-NB15 dataset website: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download the training and testing sets
3. Place them in the `data/raw/` directory

### Option 2: Use the Provided Download Script
Run the included Python script to download and preprocess the data:

```bash
python download_dataset.py
```

This script will:
- Download the dataset
- Extract and preprocess it
- Generate the processed feature files

### Option 3: Generate Processed Data from Raw Files
If you already have the raw CSV files, run:

```bash
python -c "from src.data.preprocessor import Preprocessor; p = Preprocessor(); p.process_data()"
```

## Directory Structure After Setup

```
data/
├── raw/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── processed/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── X_val.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── y_val.csv
│   ├── scaler.pkl
│   └── encoders.pkl
└── features/
    ├── selected_features.pkl
    └── mi_selector.pkl
```

## Notes

- The total dataset size is approximately 700MB when fully processed
- Preprocessing may take 10-15 minutes depending on your system
- All model checkpoints and results are included in the repository
- The `download_dataset.py` script handles all preprocessing steps automatically