# Anomaly-Based Network Intrusion Detection System

![Network Security](https://img.shields.io/badge/Network-Security-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An AI-powered security system that automatically detects cyber attacks on computer networks by learning what "normal" network traffic looks like and flagging anything suspicious.**

---

## What Is This Project?

Think of this like a **smart security guard for your network**. Traditional security systems work like a checklist—they only catch attacks they've seen before (like a guard who only stops people on a "wanted" list). This system is different: it **learns the normal patterns of your network traffic** and alerts you when something unusual happens—even if it's a brand new type of attack that nobody has ever seen before.

### Why Does This Matter?
- **Catches unknown threats**: Detects zero-day attacks and novel intrusions that signature-based systems miss
- **High accuracy**: 98.74% accuracy in identifying malicious vs. normal traffic
- **Ready to use**: Comes with pre-trained models you can deploy immediately
- **Transparent**: Full code, notebooks, and reports so you can understand and customize everything

---

## Model Comparison: How Well Does It Work?

We tested multiple AI models on real network data (257,673 connection records from the UNSW-NB15 dataset). Here's how they performed:

| Model | What It Is | Accuracy | F1-Score | Best For |
|:---|:---|:---:|:---:|:---|
| **XGBoost** ⭐ | Gradient boosting (ensemble of decision trees) | **98.74%** | **99.08%** | **Best overall — production ready** |
| **MLP (Neural Network)** | Multi-layer perceptron deep learning | 98.44% | 98.86% | Complex pattern recognition |
| **Random Forest** | Ensemble of many decision trees | 97.65% | 98.30% | Interpretable, robust baseline |
| **Logistic Regression** | Linear statistical model | 94.44% | 95.97% | Fast, simple, explainable |

![Model Performance Comparison](final_reports/model_performance_comparison.png)

> 📄 **For the full analysis, metrics breakdown, and architecture details** → [`final_reports/final_summary_report.md`](final_reports/final_summary_report.md)

---

## How It Works (In Simple Terms)

```
Network Traffic Data
        │
        ▼
┌───────────────────┐
│  CLEAN & PREP     │  → Remove errors, fill missing values, standardize formats
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  EXTRACT FEATURES │  → Calculate meaningful signals (ratios, statistics, patterns)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  SELECT BEST      │  → Keep only the 30 most informative features (mutual information)
│  FEATURES         │
└─────────┬─────────┘
          │
     ┌────┴────┐
     ▼         ▼
┌─────────┐ ┌─────────┐
│ Classical │ │ Neural  │  ← Two approaches, both trained on the same data
│   ML    │ │ Network │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │  EVALUATE   │  → Compare accuracy, precision, recall, F1-score, ROC-AUC
    │  & REPORT   │
    └─────────────┘
```

---

## What's In This Repository?

```
.
├── data/                    # Dataset info & preprocessing artifacts
│   ├── features/                   # Feature selection artifacts (.pkl)
│   └── processed/                  # Scaler, encoders (.pkl) — CSVs not stored in git
├── final_reports/           # 📊 Start here! Charts, benchmarks, executive summary
│   ├── final_summary_report.md          # Full analysis & findings
│   ├── technical_artifact.md            # System architecture & formulas
│   ├── model_performance_comparison.png # Performance charts (rendered above)
│   ├── model_performance_comparison.pdf # High-res vector charts
│   ├── model_comparison_with_mlp.csv    # All model metrics table
│   └── classical_model_comparison.csv   # Classical ML metrics
├── models/                  # 🎯 Pre-trained models (ready to use)
│   ├── best_classical_model.pkl    # XGBoost — top performer
│   └── mlp_sklearn_model.pkl       # Neural network alternative
├── notebooks/               # 📓 Step-by-step Jupyter notebooks (with outputs)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_dl_models.ipynb
└── requirements.txt         # Python dependencies
```

---

## Quick Start (3 Steps)

### 1. Install
```bash
git clone https://github.com/DeepanshuGarhkoti109/Anomaly-Based-Network-Intrusion-Detection-System.git
cd Anomaly-Based-Network-Intrusion-Detection-System
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### 2. Try the Pre-trained Model (No Training Needed)
```python
import joblib
import pandas as pd

# Load the best model (XGBoost) + preprocessing tools
model = joblib.load('models/best_classical_model.pkl')
scaler = joblib.load('data/processed/scaler.pkl')
selected_features = joblib.load('data/features/selected_features.pkl')

# Predict on your network flow data
# predictions = model.predict(your_data[selected_features])
```

### 3. Explore the Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
All notebooks are pre-executed with outputs saved — browse them directly on GitHub or open in Jupyter.

---

## Detailed Reports

| Report | What's Inside |
|:---|:---|
| [`final_summary_report.md`](final_reports/final_summary_report.md) | Executive summary, full metrics, dataset details, findings & conclusions |
| [`technical_artifact.md`](final_reports/technical_artifact.md) | System architecture, pipeline formulas, module reference, design decisions |
| [`model_comparison_with_mlp.csv`](final_reports/model_comparison_with_mlp.csv) | Raw benchmark numbers for all models (easy to import/copy) |
| [`model_performance_comparison.pdf`](final_reports/model_performance_comparison.pdf) | High-resolution vector charts for presentations & papers |

> 💡 **Tip**: Start with `final_summary_report.md` for the full story, or jump straight to the notebooks to see the code in action.

---

## Who Is This For?

- **Security teams** wanting to add ML-based anomaly detection
- **Researchers** studying network intrusion detection
- **Students** learning applied ML for cybersecurity
- **Engineers** building production NIDS pipelines

---

## License

MIT License — free to use, modify, and distribute.

---

## Citation

If you use this work in research, please cite the UNSW-NB15 dataset and this repository.