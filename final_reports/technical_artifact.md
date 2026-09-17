# Technical Overview & System Architecture

## 1. System Architecture & End-to-End Data Flow

```
UNSW-NB15.csv / NSL-KDD.txt
         │
         ▼
   DataLoader.load_raw_data()
         │
         ▼
   Preprocessor.fit_transform_pipeline()
   ├── handle_missing_values()
   ├── encode_categoricals()
   ├── clip_outliers()
   └── fit_transform_scaler()
         │
         ▼
   FeatureEngineer.run_all()
   ├── ratio features
   ├── log features
   ├── statistical features
   └── connection features
         │
         ▼
   SelectKBest(mutual_info_classif, k=30)
         │
    ┌────┴────┐
    ▼         ▼
  Classical  Neural Network / DL
  Models     Models
  (LR/RF/XGB) (MLP / DL)
    │         │
    └────┬────┘
         ▼
   evaluate_model()
   ├── accuracy, precision, recall, F1
   ├── ROC-AUC, PR-AUC
   └── confusion matrix
         │
         ▼
   final_reports/model_comparison_with_mlp.csv
```

## 2. Module Reference

### `src.data.loader.DataLoader`
| Method | Description |
|---|---|
| `load_raw_data()` | Loads UNSW-NB15 or NSL-KDD; generates synthetic data if files not found |
| `load_preprocessed_data()` | Returns `(X_train, X_test, y_train, y_test)` from `data/processed/` |

### `src.data.preprocessor.Preprocessor`
| Method | Description |
|---|---|
| `handle_missing_values(df)` | Median imputation (numeric), 'Unknown' fill (categorical) |
| `clip_outliers(df, cols, q_lo, q_hi)` | Winsorise at given quantile bounds |
| `encode_categoricals(df, cols)` | One-hot or ordinal encoding |
| `fit_scaler(X)` / `transform_scaler(X)` | StandardScaler (fit on train only) |
| `fit_transform_pipeline(df)` | All steps in sequence -> returns `(X, y, feature_names)` |

### `src.features.engineer.FeatureEngineer`
| Method | New Features |
|---|---|
| `add_ratio_features(df)` | bytes_ratio, packets_ratio, load_ratio, bytes_per_pkt |
| `add_log_features(df)` | `*_log` columns via `log1p` |
| `add_statistical_features(df)` | jitter_sum, jitter_ratio, ttl_diff, win_diff |
| `add_connection_features(df)` | ct_ratio, ct_ltm_ratio |
| `run_all(df)` | All of the above |

### `src.evaluation.metrics`
| Function | Description |
|---|---|
| `evaluate_model(y_true, y_pred, y_prob)` | Returns dict with accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| `plot_roc_curve(models_probs)` | Multi-model ROC comparison |
| `plot_confusion_matrix(y_true, y_pred)` | Annotated heatmap with FPR/FNR |
| `plot_training_history(history)` | Loss + accuracy learning curves |

### `src.utils.visualization`
| Function | Description |
|---|---|
| `plot_class_distribution(y)` | Bar + pie chart of label counts |
| `plot_correlation_matrix(df)` | Lower-triangle heatmap, reports high-corr pairs |
| `plot_feature_importance(importances)` | Horizontal bar chart |
| `plot_reconstruction_error(...)` | Autoencoder threshold visualisation |

### `src.utils.helpers`
| Function | Description |
|---|---|
| `set_random_seed(seed)` | Sets Python, NumPy, TF, and PyTorch seeds |
| `timer(label)` | Context manager that prints elapsed time |
| `timeit(func)` | Decorator version of timer |
| `memory_usage(obj)` | Human-readable RAM usage |
| `ensure_dir(path)` | `mkdir -p` wrapper |

### `src.models.trainer.ModelTrainer`
| Method | Description |
|---|---|
| `train_logistic_regression(...)` | Trains LR, saves to `models/` |
| `train_random_forest(...)` | Trains RF, saves to `models/` |
| `train_xgboost(...)` | Trains XGBoost, saves to `models/` |
| `train_all_classical(...)` | All three above, returns comparison table |
| `best_model(metric)` | Returns `(name, model)` for top model |

### `src.pipeline.run_pipeline`
```python
run_pipeline(
    dataset='unsw_nb15',    # or 'nsl_kdd'
    model='all',            # or 'xgboost', 'random_forest', 'logistic_regression'
    test_size=0.20,
    feature_engineering=True,
) -> pd.DataFrame           # comparison table
```

## 3. Mathematical Formulation & Metrics

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (True Positive Rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### ROC-AUC
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(FPR^{-1}(t)) \, dt$$

## 4. Top Selected Features (Mutual Information Ranking)

1. **sbytes** - Source-to-destination transaction bytes
2. **dbytes** - Destination-to-source transaction bytes
3. **sload** - Source bits per second
4. **dload** - Destination bits per second
5. **spkts** - Source-to-destination packet count
6. **dpkts** - Destination-to-source packet count
7. **dur** - Record total duration
8. **ct_srv_src** - Connections containing same service and source address
9. **ct_dst_sport_ltm** - Connections to same destination and source port in 100 records
10. **ct_src_dport_ltm** - Connections from same source to destination port in 100 records
