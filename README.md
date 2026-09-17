# Anomaly-Based Network Intrusion Detection System

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive AI/ML project for detecting network intrusions using anomaly-based detection techniques. This project implements multiple machine learning and deep learning approaches to identify malicious network traffic patterns.

## 📋 Project Overview

This project implements an anomaly-based Network Intrusion Detection System (NIDS) that uses machine learning to detect malicious network activities by identifying deviations from normal traffic patterns. Unlike signature-based systems, our approach learns what normal network behavior looks like and flags anomalies that could indicate security threats.

### Key Features
- **Multi-Model Approach**: Implements traditional ML, deep learning, and unsupervised anomaly detection
- **Comprehensive Evaluation**: Multiple metrics and visualization tools for model comparison
- **Real-world Datasets**: Uses UNSW-NB15 and NSL-KDD benchmark datasets
- **Modular Architecture**: Clean separation of data processing, feature engineering, and modeling
- **Reproducible Research**: Complete documentation and decision tracking

## 🏗️ Project Architecture

```
Data Acquisition → Preprocessing → Feature Engineering → Model Training → Evaluation → Reporting
```

### Model Pipeline:
1. **Data Loading**: Load and validate network traffic datasets
2. **Preprocessing**: Handle missing values, normalize features, encode labels
3. **Feature Engineering**: Extract temporal, statistical, and network-specific features
4. **Model Training**: Train multiple ML/DL models
5. **Evaluation**: Compare models using comprehensive metrics
6. **Visualization**: Generate insightful plots and reports

## 📊 Datasets

### Primary Dataset: UNSW-NB15
- **Source**: University of New South Wales
- **Size**: ~2.5 million records
- **Features**: 49 network traffic features
- **Attack Types**: 9 categories + normal traffic
- **Characteristics**: Modern attack patterns, realistic network traffic

### Secondary Dataset: NSL-KDD
- **Source**: University of New Brunswick
- **Size**: ~125,000 records
- **Features**: 41 network traffic features
- **Attack Types**: 4 main categories + normal traffic
- **Characteristics**: Well-established benchmark dataset

## 🤖 Machine Learning Models

### Traditional Machine Learning
- **XGBoost**: Gradient boosting with excellent performance
- **Random Forest**: Ensemble method for robust classification
- **Logistic Regression**: Baseline model for comparison

### Deep Learning
- **LSTM**: Recurrent neural network for temporal pattern recognition
- **Autoencoder**: Unsupervised anomaly detection
- **Multi-layer Perceptron**: Deep neural network for classification

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Precision-Recall curves
- Confusion matrices
- Feature importance analysis

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda package manager
- 8GB+ RAM recommended
- GPU optional but recommended for deep learning models

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Anomaly-Based-Network-Intrusion-Detection-System.git
cd Anomaly-Based-Network-Intrusion-Detection-System
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets (instructions in `data/README.md`)

### Usage

1. **Exploratory Data Analysis**:
```bash
jupyter notebooks/01_data_exploration.ipynb
```

2. **Run complete pipeline**:
```python
from src.pipeline import run_pipeline
results = run_pipeline(dataset='UNSW-NB15', model='all')
```

3. **Train specific model**:
```python
from src.models.trainer import ModelTrainer
trainer = ModelTrainer()
trainer.train_xgboost()
```

## 📁 Project Structure

```
.
├── data/                    # Dataset storage and utilities
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_dl_models.ipynb
├── src/                     # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model definitions and training
│   ├── utils/              # Utility functions
│   └── evaluation/         # Evaluation metrics and visualization
├── models/                  # Saved model files (.pkl)
├── final_reports/           # Final evaluation reports, technical artifacts, visualizations, and CSV metrics
├── tests/                   # Automated smoke & unit tests
├── requirements.txt         # Python dependencies
├── README.md               # Project overview
└── project_decisions.md    # Project decision log and technical journal
```

## 📈 Results

### Model Performance (UNSW-NB15 Dataset) - **ACTUAL RESULTS**
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | **98.74%** | 0.9887 | 0.9929 | **0.9908** | 0.9994 |
| **MLP (Neural Network)** | **98.44%** | 0.9845 | 0.9927 | **0.9886** | 0.9989 |
| Random Forest | 97.65% | 0.9673 | 0.9992 | 0.9830 | 0.9976 |
| Logistic Regression | 94.44% | 0.9458 | 0.9741 | 0.9597 | 0.9819 |

### **Key Findings from Completed Run**
1. **XGBoost achieved 98.74% accuracy** - Best performing classical model
2. **MLP achieved 98.44% accuracy** - Strong deep learning baseline
3. **All models >94% accuracy** - Demonstrates effective feature engineering
4. **High ROC-AUC scores** (>0.98) - Excellent discrimination capability
5. **Best model saved**: `models/best_classical_model.pkl` (XGBoost)
6. **Complete report**: `final_reports/final_summary_report.md`

### **Project Completion Status** ✅
- ✅ Dataset downloaded and verified
- ✅ All 5 notebooks executed successfully  
- ✅ 4 ML/DL models trained and evaluated
- ✅ Best models saved and validated
- ✅ Comprehensive report & charts generated in `final_reports/`

*Results based on actual run completed on September 16, 2026*

## 🔧 Configuration

The project can be configured through `config.yaml`:
```yaml
datasets:
  unsw_nb15:
    path: data/UNSW-NB15.csv
    train_split: 0.7
    test_split: 0.15
    val_split: 0.15
  
models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  
  lstm:
    units: 128
    dropout: 0.2
    epochs: 50
    batch_size: 64
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_data.py     # Data processing tests
pytest tests/test_models.py   # Model tests
pytest tests/test_utils.py    # Utility function tests
```

## 📝 Documentation

- **Technical Documentation**: `docs/technical/`
- **API Reference**: `docs/api/`
- **User Guide**: `docs/user_guide.md`
- **Decision Log**: `project_decisions.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- University of New South Wales for the UNSW-NB15 dataset
- University of New Brunswick for the NSL-KDD dataset
- The open-source ML community for libraries and tools
- Academic researchers in network security and machine learning

## 📞 Contact

**Deepanshu Garhkoti**  
Project Lead  
Email: deepanshu@example.com  
GitHub: [@deepanshugarhkoti](https://github.com/deepanshugarhkoti)

---

*This project was recreated on September 15, 2026, after the original repository was lost. All decisions and implementation details are documented in `project_decisions.md`.*


---

## 🎯 **PROJECT COMPLETION STATUS: SUCCESSFULLY EXECUTED** ✅

**All tasks completed successfully on September 16, 2026**

### **What Was Accomplished:**
1. **✅ Dataset Acquisition**: UNSW-NB15 training (30.8 MB) and testing (14.7 MB) sets downloaded
2. **✅ Data Pipeline**: Complete EDA → Preprocessing → Feature Engineering pipeline executed
3. **✅ Model Training**: Logistic Regression, Random Forest, and XGBoost models trained
4. **✅ Model Evaluation**: Comprehensive performance comparison with metrics
5. **✅ Best Model Saved**: XGBoost classifier (98.74% accuracy) saved as `best_classical_model.pkl`
6. **✅ Final Report**: Complete documentation with visualizations in `final_report/`

### **Quick Start - Use the Trained Model:**
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/best_classical_model.pkl')

# Load preprocessing tools
scaler = joblib.load('data/processed/scaler.pkl')
selected_features = joblib.load('data/features/selected_features.pkl')

# Make predictions on new data
predictions = model.predict(new_data[selected_features])
```

### **View Results:**
- **Performance Report**: `final_reports/final_summary_report.md`
- **Model Comparison Table**: `final_reports/model_comparison_with_mlp.csv`
- **Visualizations**: `final_reports/model_performance_comparison.png`
- **Executed Notebooks**: `notebook_outputs/`

---


## ⚠️ Important Note: Large Data Files

Due to GitHub's file size limitations (100MB per file), the processed data files are not included in this repository. Please follow the instructions in [DATA_DOWNLOAD_INSTRUCTIONS.md](DATA_DOWNLOAD_INSTRUCTIONS.md) to download and set up the data.

### Quick Data Setup:
```bash
# Option 1: Run the download script
python download_dataset.py

# Option 2: If you have the raw files, run preprocessing
python run_notebooks.py --notebook 02_preprocessing
```

### Files Excluded from Git:
- `data/processed/X_train.csv` (473 MB)
- `data/processed/X_test.csv` (101 MB) 
- `data/processed/X_val.csv` (101 MB)
- `data/raw/UNSW_NB15_training-set.csv` (30.8 MB)
- `data/raw/UNSW_NB15_testing-set.csv` (14.7 MB)

All other files including trained models, results, and documentation are included.