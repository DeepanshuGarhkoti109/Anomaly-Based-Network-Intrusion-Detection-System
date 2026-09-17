# Anomaly-Based Network Intrusion Detection System

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A production-grade AI/ML system for detecting network intrusions using anomaly-based detection techniques. This repository contains the complete modular pipeline, interactive Jupyter notebooks, pre-trained models, and comprehensive benchmark reports evaluating classical machine learning (XGBoost, Random Forest, Logistic Regression) and neural networks (MLP).

---

## 📋 Project Overview

This project implements an anomaly-based Network Intrusion Detection System (NIDS) that detects malicious network traffic by identifying deviations from normal connection patterns. Unlike signature-based systems that rely solely on known attack rules, this system learns normal network behavior patterns to detect zero-day vulnerabilities and novel intrusions.

### Key Highlights
- **High Detection Performance**: **98.74% Accuracy** and **99.08% F1-Score** achieved using tuned XGBoost.
- **End-to-End Pipeline**: Modular data ingestion, Winsorization, standardization, one-hot encoding, and mutual-information feature selection.
- **Pre-trained Serialized Models**: XGBoost classifier and Multi-Layer Perceptron (MLP) weights ready for immediate inference.
- **Executed Jupyter Notebooks & HTML Reports**: Clean, verified step-by-step notebooks (01 to 05) with exported interactive HTML reports.

---

## 🏗️ System Architecture & Data Flow

```
UNSW-NB15 Raw Flow Records
         │
         ▼
   DataLoader.load_raw_data()
         │
         ▼
   Preprocessor.fit_transform_pipeline()
   ├── Median imputation & categorical encoding
   ├── Outlier clipping (Winsorization)
   └── StandardScaler normalization
         │
         ▼
   FeatureEngineer.run_all()
   ├── Ratio & log-transformed features
   ├── Statistical aggregation & connection features
   └── SelectKBest (Mutual Information ranking, k=30)
         │
    ┌────┴────┐
    ▼         ▼
  Classical  Neural Networks
  ML Models  (Multi-Layer Perceptron)
  (LR/RF/XGB)
    │         │
    └────┬────┘
         ▼
   evaluate_model()
   └── final_reports/ (Markdown summary, charts & CSV benchmarks)
```

---

## 📈 Model Performance & Benchmark Results

Evaluated on the UNSW-NB15 benchmark test dataset (257,673 total records):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Best Model)** | **98.74%** | **0.9887** | **0.9929** | **0.9908** | **0.9994** |
| **MLP (Neural Network)** | **98.44%** | 0.9845 | 0.9927 | **0.9886** | **0.9989** |
| **Random Forest** | 97.65% | 0.9673 | **0.9992** | 0.9830 | 0.9976 |
| **Logistic Regression** | 94.44% | 0.9458 | 0.9741 | 0.9597 | 0.9819 |

> 📊 Detailed analysis and visualizations can be found in [`final_reports/final_summary_report.md`](final_reports/final_summary_report.md) and [`final_reports/technical_artifact.md`](final_reports/technical_artifact.md).

---

## 📁 Repository Structure

```
.
├── data/                    # Dataset documentation and preprocessing artifacts
│   ├── features/           # Feature selection artifacts (mi_selector.pkl, selected_features.pkl)
│   ├── processed/          # Preprocessing artifacts (scaler.pkl, encoders.pkl)
│   └── README.md           # Dataset download instructions and citations
├── final_reports/           # Consolidated reports, technical documentation & benchmarks
│   ├── final_summary_report.md          # Comprehensive executive summary & analysis
│   ├── technical_artifact.md            # System architecture, module reference & formulas
│   ├── model_comparison_with_mlp.csv    # Full model benchmark table
│   ├── classical_model_comparison.csv   # Classical ML metrics table
│   ├── model_performance_comparison.png # High-res comparison plots
│   └── model_performance_comparison.pdf # Vector comparison plots
├── models/                  # Pre-trained serialized model weights (.pkl)
│   ├── best_classical_model.pkl         # Trained XGBoost model
│   └── mlp_sklearn_model.pkl            # Trained Neural Network (MLP) model
├── notebooks/               # Step-by-step Jupyter notebooks
│   ├── 01_data_exploration.ipynb        # Exploratory Data Analysis & visual profiling
│   ├── 02_preprocessing.ipynb           # Cleaning, imputation, scaling & encoding
│   ├── 03_feature_engineering.ipynb     # Feature extraction & mutual information selection
│   ├── 04_ml_models.ipynb               # Classical ML model training & evaluation
│   └── 05_dl_models.ipynb               # Deep learning & MLP neural network models
├── notebook_outputs/        # Pre-executed notebooks (.ipynb) and HTML export views (.html)
├── src/                     # Core Python modules & pipeline
│   ├── data/               # DataLoader and Preprocessor classes
│   ├── features/           # FeatureEngineer class
│   ├── models/             # ModelTrainer class
│   ├── evaluation/         # Metrics evaluation and visualization helpers
│   ├── utils/              # Helper utilities and visualization routines
│   └── pipeline.py         # End-to-end automated pipeline entrypoint
├── requirements.txt         # Python package dependencies
└── README.md               # Project overview and guide
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/DeepanshuGarhkoti109/Anomaly-Based-Network-Intrusion-Detection-System.git
cd Anomaly-Based-Network-Intrusion-Detection-System

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # On Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Using the Pre-trained Model for Inference

```python
import joblib
import pandas as pd

# 1. Load trained XGBoost model and preprocessors
model = joblib.load('models/best_classical_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')
selected_features = joblib.load('data/features/selected_features.pkl')

# 2. Predict on new network flows
# predictions = model.predict(new_flow_features[selected_features])
```

### 3. Running the Pipeline or Notebooks

You can run the end-to-end pipeline programmatically:
```python
from src.pipeline import run_pipeline

results = run_pipeline(dataset='unsw_nb15', model='all')
print(results)
```

Or open and run the interactive notebooks:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📄 License
This project is licensed under the MIT License.