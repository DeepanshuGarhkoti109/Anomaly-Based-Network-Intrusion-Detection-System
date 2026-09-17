# Dataset Information

This directory contains datasets used for the Anomaly-Based Network Intrusion Detection System project.

## Primary Dataset: UNSW-NB15

### Download Instructions
1. Visit the official UNSW-NB15 dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download the CSV files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
3. Place the files in this directory
4. Run the data preparation script:
   ```bash
   python src/data/prepare_unsw_nb15.py
   ```

### Dataset Structure
The UNSW-NB15 dataset contains:
- **Records**: Approximately 2.5 million network traffic records
- **Features**: 49 features describing network connections
- **Attack Types**: 9 categories + normal traffic
- **Format**: CSV with headers

### Feature Categories
1. **Basic Features**: srcip, sport, dstip, dport, proto, state
2. **Content Features**: service, duration, srcbytes, dstbytes
3. **Time-based Features**: stime, ltime, sinpkt, dinpkt, sjit, djit
4. **Connection Features**: swin, dwin, stcpb, dtcpb, smeansz, dmeansz
5. **Statistical Features**: ct_state_ttl, ct_flw_http_mthd, ct_srv_src, ct_srv_dst

### Attack Categories
1. **Normal**: Benign network traffic
2. **Analysis**: Port scan, vulnerability scan
3. **Backdoor**: Malware, trojan horse
4. **DoS**: Denial of Service attacks
5. **Exploits**: Vulnerability exploitation
6. **Fuzzers**: Fuzzing attacks
7. **Generic**: Generic attacks
8. **Reconnaissance**: Information gathering
9. **Shellcode**: Code injection attacks
10. **Worms**: Self-replicating malware

## Secondary Dataset: NSL-KDD

### Download Instructions
1. Visit the NSL-KDD dataset page: https://www.unb.ca/cic/datasets/nsl.html
2. Download:
   - `KDDTrain+.txt`
   - `KDDTest+.txt`
3. Place the files in this directory
4. Run the data preparation script:
   ```bash
   python src/data/prepare_nsl_kdd.py
   ```

### Dataset Structure
The NSL-KDD dataset contains:
- **Records**: Approximately 125,000 network traffic records
- **Features**: 41 features (including labels)
- **Attack Types**: 4 main categories + normal traffic
- **Format**: Text files without headers

### Feature Categories
1. **Basic Features**: duration, protocol_type, service, flag, src_bytes, dst_bytes
2. **Content Features**: land, wrong_fragment, urgent, hot
3. **Time-based Features**: count, srv_count, serror_rate, srv_serror_rate
4. **Host Features**: same_srv_rate, diff_srv_rate, srv_diff_host_rate
5. **Statistical Features**: dst_host_count, dst_host_srv_count, dst_host_same_srv_rate

### Attack Categories
1. **Normal**: Benign network traffic
2. **DoS**: Denial of Service
3. **Probe**: Surveillance and probing
4. **R2L**: Unauthorized access from remote
5. **U2R**: Unauthorized access to root

## Data Preparation Notes

### Preprocessing Steps
1. **Missing Values**: Handle missing values using median imputation
2. **Feature Scaling**: Standardize numerical features
3. **Categorical Encoding**: One-hot encode categorical features
4. **Label Encoding**: Convert attack types to numerical labels
5. **Train-Test Split**: Split data into training, validation, and test sets

### File Structure After Preparation
```
data/
├── raw/                    # Original downloaded files
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── processed/              # Preprocessed data
│   ├── unsw_nb15_train.csv
│   ├── unsw_nb15_test.csv
│   ├── unsw_nb15_val.csv
│   ├── nsl_kdd_train.csv
│   ├── nsl_kdd_test.csv
│   └── nsl_kdd_val.csv
└── features/              # Feature-engineered data
    ├── unsw_nb15_features.pkl
    └── nsl_kdd_features.pkl
```

### Usage in Code
```python
from src.data.loader import DataLoader

# Load UNSW-NB15 dataset
loader = DataLoader(dataset='unsw_nb15')
X_train, X_test, y_train, y_test = loader.load_data()

# Load with preprocessing
X_train_processed, X_test_processed, y_train, y_test = loader.load_preprocessed_data()
```

## Dataset Citations

### UNSW-NB15 Citation
```
Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). 
In 2015 military communications and information systems conference (MilCIS) (pp. 1-6). IEEE.
```

### NSL-KDD Citation
```
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. 
In 2009 IEEE symposium on computational intelligence for security and defense applications (pp. 1-6). IEEE.
```

## Notes
- Ensure you have sufficient disk space (approx. 5GB for raw data)
- The preprocessing scripts will create additional processed files
- Always verify data integrity after download
- Keep original files in the `raw/` directory for reproducibility