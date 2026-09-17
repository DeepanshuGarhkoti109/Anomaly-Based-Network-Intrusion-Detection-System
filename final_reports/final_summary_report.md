# Final Evaluation Report: Anomaly-Based Network Intrusion Detection System

**Report Generated:** 2026-09-17 14:38:47  
**Dataset:** UNSW-NB15  
**Total Records Evaluated:** 257,673 (175,341 train + 82,332 test)  

## 1. Executive Summary

Four machine learning and neural network models were evaluated on the UNSW-NB15 network intrusion detection benchmark. The **XGBoost** classifier achieved the highest overall detection score with **98.74% accuracy** and **99.08% F1-score**, demonstrating robust intrusion detection capability across various cyber attack families.

## 2. Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.9444 (94.44%) | 0.9458 | 0.9741 | 0.9597 | 0.9819 |
| **Random Forest** | 0.9765 (97.65%) | 0.9673 | 0.9992 | 0.9830 | 0.9976 |
| **XGBoost** | 0.9874 (98.74%) | 0.9887 | 0.9929 | 0.9908 | 0.9994 |
| **MLP (scikit-learn)** | 0.9844 (98.44%) | 0.9845 | 0.9927 | 0.9886 | 0.9989 |

### Key Findings

1. **Top Classical Performer:** XGBoost achieved top accuracy (98.74%) with an ROC-AUC of 0.9994.
2. **Neural Network Capabilities:** Multi-Layer Perceptron (MLP) achieved strong performance (98.44% accuracy), closely matching gradient boosting.
3. **High Discriminative Power:** All models achieved ROC-AUC > 0.98, confirming effective feature representation and pre-filtering.
4. **High Recall on Threats:** XGBoost and Random Forest achieved >99% recall on malicious flows, minimizing undetected intrusions.

## 3. Pipeline & Technical Details

### Dataset Specifications
- **Benchmark:** UNSW-NB15
- **Network Traffic Features:** 49 base flow and content attributes
- **Attack Types Detected:** 9 categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) + Normal

### Preprocessing & Feature Engineering
1. **Data Cleaning:** Median imputation and validation for missing values
2. **Feature Scaling:** Standardization using `StandardScaler`
3. **Categorical Encoding:** One-hot encoding of protocol and state features
4. **Feature Selection:** Mutual Information (`SelectKBest` with mutual info score) selecting top 30 features

### Model Architecture & Hyperparameters

#### 1. XGBoost Classifier (Best Model)
- `n_estimators`: 100
- `learning_rate`: 0.1
- `max_depth`: 6
- `subsample`: 0.8
- `colsample_bytree`: 0.8

#### 2. Multi-Layer Perceptron (MLP)
- `hidden_layer_sizes`: (256, 128, 64)
- `activation`: ReLU
- `solver`: Adam
- `batch_size`: 256
- `early_stopping`: True

#### 3. Random Forest Classifier
- `n_estimators`: 100
- `criterion`: Gini impurity
- `random_state`: 42

#### 4. Logistic Regression
- `penalty`: L2
- `solver`: liblinear
- `max_iter`: 1000

## 4. Consolidated Artifacts in `final_reports/`

| File | Description |
|:---|:---|
| [`final_summary_report.md`](./final_summary_report.md) | Comprehensive evaluation and analysis report |
| [`technical_artifact.md`](./technical_artifact.md) | Technical overview, module reference & mathematical formulation |
| [`model_performance_comparison.png`](./model_performance_comparison.png) | High-resolution performance visualization chart |
| [`model_performance_comparison.pdf`](./model_performance_comparison.pdf) | Vector graphics performance chart |
| [`model_comparison_with_mlp.csv`](./model_comparison_with_mlp.csv) | Full metrics table (Classical + MLP) |
| [`classical_model_comparison.csv`](./classical_model_comparison.csv) | Classical ML benchmarks table |

## 5. Conclusion & Deployment

The intrusion detection pipeline achieves production-grade accuracy (98.74%) with sub-millisecond inference time per flow. All artifacts and benchmark data are fully reproducible and consolidated within the `final_reports/` directory.
