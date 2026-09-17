# Technical Artifact: Model Evaluation Details

## Model Configuration Details

### Hyperparameters
```python
# Model hyperparameters used in training
model_configs = {
    'LogisticRegression': {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42
    },
    'RandomForestClassifier': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'XGBClassifier': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
}
```

## Evaluation Metrics Formulas

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision
```
Precision = TP / (TP + FP)
```

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```

### F1 Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### ROC-AUC
```
ROC-AUC = Area under the Receiver Operating Characteristic curve
```

## Feature Importance Analysis

The top 10 most important features selected by mutual information:

1. **sbytes** - Source bytes
2. **dbytes** - Destination bytes
3. **sload** - Source load
4. **dload** - Destination load
5. **spkts** - Source packets
6. **dpkts** - Destination packets
7. **dur** - Duration
8. **ct_srv_src** - Connection count per service-source
9. **ct_dst_sport_ltm** - Connection count per destination-sport
10. **ct_src_dport_ltm** - Connection count per source-dport

## Statistical Significance

The performance difference between XGBoost and other models is statistically significant (p < 0.05) based on paired t-tests of prediction confidence scores.
