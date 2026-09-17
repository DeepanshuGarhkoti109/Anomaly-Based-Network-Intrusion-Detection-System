# Anomaly-Based Network Intrusion Detection System
# Project Decision Log & Technical Journal

**Author:** Deepanshu Garhkoti (A2305220519)  
**College:** [Your College Name]  
**Date Recreated:** September 15, 2026  
**Original Draft:** 4th Draft  

---

## Table of Contents
1. [Project Background](#1-project-background)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Decisions](#3-dataset-decisions)
4. [Architecture Decisions](#4-architecture-decisions)
5. [Preprocessing Decisions](#5-preprocessing-decisions)
6. [Feature Engineering Decisions](#6-feature-engineering-decisions)
7. [Model Selection Decisions](#7-model-selection-decisions)
8. [Deep Learning Design Decisions](#8-deep-learning-design-decisions)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [Project Structure Decisions](#10-project-structure-decisions)
11. [Technical Stack Decisions](#11-technical-stack-decisions)
12. [Implementation Notes](#12-implementation-notes)
13. [Known Limitations](#13-known-limitations)
14. [Future Work](#14-future-work)
15. [References](#15-references)

---

## 1. Project Background

### Why Anomaly-Based Detection?

Network intrusion detection systems (NIDS) fall into two broad categories:

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Signature-based** | Matches traffic against known attack patterns | Fast, low false positives | Misses zero-day attacks, needs constant updates |
| **Anomaly-based** | Learns "normal" behaviour, flags deviations | Detects unknown attacks | Higher false positive rate, needs training data |

**Decision:** We implement anomaly-based detection because:
- It can detect novel, previously unseen attacks
- It is more aligned with modern ML/AI research directions
- It demonstrates a richer set of ML techniques (supervised + unsupervised)
- It is the approach featured in the original project draft

### What We Are Detecting

The system classifies network traffic flows as:
- **Binary:** Normal (0) vs Attack (1)
- **Multi-class:** Normal + 9 attack categories (DoS, Exploits, Reconnaissance, etc.)

---

## 2. Problem Statement

> *"Given a set of features describing a network traffic flow, determine whether the flow represents normal behaviour or a malicious intrusion attempt."*

### Challenges
1. **Class imbalance** — normal traffic typically outnumbers attacks
2. **High dimensionality** — 49 features in UNSW-NB15
3. **Categorical features** — protocol type, service, state
4. **Temporal patterns** — attacks often span multiple sequential flows
5. **Unknown attacks** — model must generalise beyond training examples

---

## 3. Dataset Decisions

### Decision 3.1 — Primary Dataset: UNSW-NB15

**Chosen:** UNSW-NB15  
**Why:**
- Modern dataset (2015) reflecting realistic current attack patterns
- 49 well-documented features across 5 categories
- 9 distinct attack categories + normal traffic
- Widely used in academic benchmarks — allows comparison with published results
- Maintained by UNSW Canberra with active community support

**Key Statistics:**
| Attribute | Value |
|---|---|
| Total records | ~2.5 million |
| Features | 49 |
| Attack categories | 9 + Normal |
| Normal traffic | ~56.6% |
| Attack traffic | ~43.4% |

**Citation:**  
Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS 2015*. IEEE.

### Decision 3.2 — Secondary Dataset: NSL-KDD

**Chosen:** NSL-KDD as a secondary benchmark  
**Why:**
- Classic benchmark — allows comparison with older literature
- Smaller size makes it ideal for rapid experimentation
- Good for validating that our pipeline is dataset-agnostic
- 4 main attack categories (DoS, Probe, R2L, U2R)

**Why not CIC-IDS-2017:**
- 51 GB raw size is impractical for academic project hardware
- Requires packet-capture tools (CICFlowMeter) to reproduce
- UNSW-NB15 already covers the key attack types

### Decision 3.3 — Synthetic Fallback Data

**Decision:** Implemented a synthetic data generator in `DataLoader` that produces a realistic 10,000-row dataset when the real CSV files are not present.

**Why:** Enables the entire codebase to run, all notebooks to be demonstrated, and smoke tests to pass even without the (large) real dataset downloaded. The synthetic data matches the real data's column schema and approximate class distribution.

**Activation:** The synthetic data is generated automatically with a warning message pointing to the official download URL.

---

## 4. Architecture Decisions

### Decision 4.1 — Modular Pipeline Architecture

**Chosen architecture:**
```
DataLoader → Preprocessor → FeatureEngineer → ModelTrainer → Evaluator
```

**Why modular:**
- Each component is independently testable
- Notebooks can skip/re-run individual steps without re-running the whole pipeline
- Enables easy swapping of individual components (e.g., try a different scaler)
- Matches industry best practices for reproducible ML

### Decision 4.2 — Notebook-First with Supporting Modules

**Decision:** Primary work lives in Jupyter notebooks; reusable logic is extracted into `src/` Python modules.

**Why:**
- Notebooks provide visual, step-by-step documentation suitable for academic presentation
- Extracting logic into modules avoids code duplication across notebooks
- Notebooks import from `src/` making them thin orchestration layers
- This is the standard pattern in data science projects (see: cookiecutter-datascience)

### Decision 4.3 — Five-Notebook Structure

| Notebook | Purpose | Key Output |
|---|---|---|
| 01_data_exploration | EDA, quality checks, visualisations | Insight report |
| 02_preprocessing | Cleaning, encoding, scaling, splitting | Processed CSV files |
| 03_feature_engineering | MI selection, PCA visualisation | `selected_features.pkl` |
| 04_ml_models | LR, RF, XGBoost training + comparison | `best_classical_model.pkl` |
| 05_dl_models | MLP, LSTM, Autoencoder | Keras `.keras` files |

**Why five notebooks instead of one:** Separating concerns makes it easy to re-run a single stage without re-executing everything. This is critical when training deep learning models (notebook 05) which take significantly longer than the others.

---

## 5. Preprocessing Decisions

### Decision 5.1 — Median Imputation for Missing Values

**Chosen:** `SimpleImputer(strategy='median')` for numeric columns  
**Why:**
- Median is robust to outliers (unlike mean)
- Network traffic data is heavily right-skewed (a few extremely large values)
- Categorical columns filled with "Unknown" to preserve cardinality

**Alternative considered:** Drop rows with missing values  
**Why not:** Could remove too many samples; median imputation is standard practice

### Decision 5.2 — Percentile Clipping for Outliers (1st–99th percentile)

**Chosen:** `df[col].clip(lower=q1, upper=q99)`  
**Why:**
- Winsorising at 1st/99th percentile retains the shape of the distribution while eliminating extreme values
- Network traffic contains legitimate extreme values (e.g., large file transfers) that should not be discarded entirely
- Preferred over z-score removal because it is deterministic and doesn't lose rows

### Decision 5.3 — One-Hot Encoding for Categorical Features

**Chosen:** `pd.get_dummies(..., drop_first=True)`  
**Why:**
- Protocol type (tcp/udp/icmp), service (http/ftp/...), and state (FIN/CON/...) are nominal — no ordinal relationship
- Tree-based models (RF, XGBoost) can technically handle ordinal encoding but one-hot is cleaner
- `drop_first=True` avoids dummy variable trap (multicollinearity)

### Decision 5.4 — StandardScaler

**Chosen:** `StandardScaler` (zero mean, unit variance)  
**Why:**
- Required for Logistic Regression and neural networks to converge properly
- Does not compress the range like MinMaxScaler (better for outlier-clipped data)
- **Critical:** Scaler is fit only on training data, then applied to val/test — prevents data leakage

### Decision 5.5 — 70/15/15 Train/Val/Test Split

**Chosen:** 70% train, 15% validation, 15% test  
**Why:**
- Sufficient training data for all model types including deep learning
- Separate validation set used for early stopping in neural networks
- Test set never seen during training or hyperparameter tuning
- Stratified split preserves class distribution in all three sets

---

## 6. Feature Engineering Decisions

### Decision 6.1 — Mutual Information for Feature Selection

**Chosen:** `SelectKBest(mutual_info_classif, k=30)`  
**Why:**
- MI captures non-linear relationships (unlike Pearson correlation which only captures linear)
- Model-agnostic — works for both tree-based and neural network models
- Reduces dimensionality from ~35+ features to 30 most informative ones
- Reduces training time and risk of overfitting

**Alternative considered:** PCA for dimensionality reduction  
**Why not as primary selection:** PCA creates new axes that are linear combinations of features — this loses interpretability. MI selection retains the original, interpretable features.

### Decision 6.2 — Engineered Features

The following domain-specific features were added:

| Feature | Formula | Rationale |
|---|---|---|
| `bytes_ratio` | sbytes / (sbytes + dbytes + 1) | Directional asymmetry in traffic |
| `packets_ratio` | spkts / (spkts + dpkts + 1) | Packet flow imbalance |
| `load_ratio` | sload / (sload + dload + 1) | Load asymmetry |
| `bytes_per_pkt` | (sbytes+dbytes) / (spkts+dpkts+1) | Average packet size |
| `jitter_sum` | sjit + djit | Total network jitter |
| `jitter_ratio` | sjit / (sjit+djit+ε) | Directional jitter imbalance |
| `ttl_diff` | abs(sttl - dttl) | TTL hop asymmetry |
| `win_diff` | abs(swin - dwin) | TCP window asymmetry |
| `*_log` | log1p(col) | Normalise skewed byte/packet cols |

**Rationale:** Attack traffic often exhibits strong asymmetry — e.g., a DoS attack sends many packets but receives few responses. Ratio features capture this directly.

---

## 7. Model Selection Decisions

### Decision 7.1 — Three-Tier Approach

We implemented three distinct tiers to demonstrate breadth and enable comparison:

```
Tier 1 (Baseline)    : Logistic Regression
Tier 2 (Classical ML): Random Forest + XGBoost  
Tier 3 (Deep Learning): MLP + LSTM + Autoencoder
```

### Decision 7.2 — Logistic Regression as Baseline

**Why include it:**
- Provides a minimum performance floor — any useful model must beat this
- Fast to train, easy to interpret coefficients
- Good for checking that feature scaling is working correctly (LR is sensitive to unscaled features)

### Decision 7.3 — Random Forest

**Configuration:** 100 estimators, max_depth=10, balanced class weights  
**Why:**
- Strong ensemble method with built-in feature importance
- Robust to outliers and non-linear relationships
- Relatively fast training even on large datasets
- `max_depth=10` limits overfitting without sacrificing accuracy

### Decision 7.4 — XGBoost as Primary Classical Model

**Configuration:** 100 estimators, max_depth=6, learning_rate=0.1, subsample=0.8  
**Why XGBoost:**
- Consistently top performer in tabular data competitions (Kaggle)
- Built-in regularisation (L1/L2) prevents overfitting
- Native handling of missing values (not needed here but good practice)
- Fastest among gradient boosting implementations
- Provides SHAP-compatible feature importance

**Why these hyperparameters:**
- `max_depth=6`: sweet spot — deep enough to capture interactions, not so deep it overfits
- `learning_rate=0.1`: standard starting point with 100 estimators
- `subsample=0.8`: introduces randomness, reduces variance

---

## 8. Deep Learning Design Decisions

### Decision 8.1 — MLP Architecture

```
Input(n_features) → Dense(256, ReLU) → BN → Dropout(0.3)
                  → Dense(128, ReLU) → BN → Dropout(0.3)
                  → Dense(64, ReLU) → Dense(1, Sigmoid)
```

**Why this design:**
- BatchNormalization stabilises training, allows higher learning rates
- Dropout(0.3) prevents overfitting on large dataset
- Decreasing layer sizes (256→128→64) forms a bottleneck that forces feature compression
- Single sigmoid output for binary classification

### Decision 8.2 — LSTM for Temporal Modelling

**Design:** Sliding window of 5 consecutive flows → LSTM(128) → LSTM(64) → Dense(1)

**Why LSTM:**
- Network attacks often span multiple sequential connections (e.g., port scanning)
- LSTM can capture temporal dependencies that are invisible to single-flow models
- Bidirectional LSTM considered but adds training complexity not needed for this scope

**Why window size 5:**
- Small enough to be computationally feasible
- Large enough to capture multi-step attack patterns (e.g., SYN flood sequences)
- Padding with zeros for the first `timesteps-1` samples preserves row count

### Decision 8.3 — Autoencoder for Unsupervised Detection

```
Input → Dense(64) → Dense(32) → Dense(16, bottleneck) 
      → Dense(32) → Dense(64) → Output(reconstruction)
```

**Key design choice:** Trained **only on normal traffic**  
**Why:** The autoencoder learns to reconstruct normal patterns well. Attack traffic, being structurally different, produces high reconstruction error. We classify as "attack" any sample whose MSE exceeds the 95th percentile of normal reconstruction errors.

**Why 95th percentile as threshold:**
- Allows 5% false positives on normal traffic (acceptable sensitivity/specificity tradeoff)
- Data-driven — adapts to the actual distribution of normal traffic
- Can be tuned up/down depending on the operational requirement (higher security vs. fewer alerts)

### Decision 8.4 — Early Stopping

Applied to all neural networks:  
- `monitor='val_loss'`, `patience=10`, `restore_best_weights=True`

**Why:** Prevents overfitting without manual epoch tuning. `restore_best_weights=True` ensures we use the best checkpoint, not the last epoch.

---

## 9. Evaluation Strategy

### Decision 9.1 — Primary Metric: F1 Score

**Why F1 over Accuracy:**
- Class imbalance means a model that predicts "all normal" can still achieve >56% accuracy on UNSW-NB15
- F1 score balances precision and recall, penalising both false positives and false negatives
- In security context, false negatives (missed attacks) are especially costly

### Decision 9.2 — Full Metric Suite

| Metric | What it measures | Why included |
|---|---|---|
| Accuracy | Overall correct predictions | Baseline understanding |
| Precision | Of all "attack" predictions, how many were real? | Operational alert quality |
| Recall | Of all real attacks, how many were caught? | Security coverage |
| F1 Score | Harmonic mean of precision and recall | Primary model comparison metric |
| ROC-AUC | Ranking quality across all thresholds | Threshold-independent evaluation |
| PR-AUC | Precision-recall tradeoff | Better than ROC-AUC under imbalance |

### Decision 9.3 — Confusion Matrix Reporting

Always reported alongside metrics to show:
- False Positive Rate (FPR): legitimate traffic flagged as attack → operator burden
- False Negative Rate (FNR): attacks missed → security risk

---

## 10. Project Structure Decisions

### Decision 10.1 — Directory Layout

```
project/
├── data/               Raw → processed → features (three-stage pipeline)
├── notebooks/          01–05 execution sequence
├── src/                Importable Python modules
│   ├── data/           DataLoader, Preprocessor
│   ├── features/       FeatureEngineer
│   ├── models/         ModelTrainer
│   ├── evaluation/     metrics, plots
│   └── utils/          visualization, helpers
├── models/             Saved .pkl / .keras files
├── results/            CSV comparison tables
├── tests/              smoke_test.py
├── docs/               Extended documentation
├── config.yaml         All hyperparameters in one place
├── requirements.txt    Pinned dependencies
└── generate_notebooks.py  Notebook source-of-truth (avoids JSON errors)
```

**Why `generate_notebooks.py`:**
Writing `.ipynb` files as raw JSON strings in a text editor is error-prone — any unescaped character breaks the JSON. Instead, all notebook content is defined as Python lists of strings and serialised with `json.dumps`. This guarantees valid JSON and makes the notebooks easy to update programmatically.

### Decision 10.2 — `config.yaml` as Central Configuration

All hyperparameters (dataset paths, model params, split ratios, etc.) live in one `config.yaml` file. This means:
- No "magic numbers" scattered through notebooks and modules
- Easy to reproduce exact experiments
- A reviewer can see all configuration at a glance

---

## 11. Technical Stack Decisions

### Decision 11.1 — Python 3.8+

**Why:** Broad ecosystem support, type hints, walrus operator. All major ML libraries support 3.8+.

### Decision 11.2 — scikit-learn for Classical ML

**Why:**
- Industry standard, well-documented
- Consistent `fit/predict/predict_proba` API across all classical models
- Extensive preprocessing utilities (`StandardScaler`, `SelectKBest`, etc.)
- `Pipeline` object for clean composition

### Decision 11.3 — XGBoost (standalone, not sklearn wrapper)

**Why:** Native XGBoost API gives access to early stopping, SHAP values, and custom objectives not available through the sklearn wrapper.

### Decision 11.4 — TensorFlow/Keras for Deep Learning

**Why:**
- Most widely used deep learning framework in academic settings
- High-level Keras API makes model definition concise
- Strong GPU support (optional — models work on CPU too)
- Excellent documentation and community

**Alternative considered:** PyTorch  
**Why not:** Keras is more accessible for academic audiences and the project doesn't require PyTorch-specific features.

### Decision 11.5 — joblib for Model Serialisation

**Why:** `joblib` is faster than `pickle` for large NumPy arrays (as used inside sklearn/XGBoost models). It is the serialisation method recommended by scikit-learn.

---

## 12. Implementation Notes

### Synthetic Data Fallback

The `DataLoader` automatically generates a 10,000-row synthetic dataset when the real UNSW-NB15 CSV files are not found. This allows:
- All 5 notebooks to run without downloading data
- Smoke tests to run in CI/CD environments
- Quick demonstration of the pipeline

The synthetic data is seeded (`np.random.default_rng(42)`) for reproducibility and matches the real data's schema and approximate class distribution (57% normal / 43% attack).

### Data Leakage Prevention

The following measures prevent data leakage:
1. `StandardScaler` is **fit only on the training set**, then applied to val/test
2. Feature selection (`SelectKBest`) is **fit only on the training set**
3. The Autoencoder is **trained only on normal training samples**
4. All random splits use `stratify=y` to maintain class ratios

### Logging

All modules use `logging` (not `print`) so output can be suppressed or redirected. The default level is `INFO` — set to `DEBUG` for verbose output.

---

## 13. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Synthetic data used in tests | Results don't reflect real performance | Download UNSW-NB15 for real evaluation |
| No hyperparameter optimisation | Models use sensible defaults, not optimal params | Add `GridSearchCV` / Optuna in future |
| LSTM uses zero-padding | First `timesteps-1` samples have no real history | Acceptable for demo; use stateful LSTM in production |
| No SMOTE / resampling | Class imbalance handled by metrics only | Add `imbalanced-learn` SMOTE in future |
| No real-time inference module | Batch prediction only | Add streaming inference with Kafka/FastAPI |
| TensorFlow not tested on all configs | GPU/CPU compatibility varies | Use `tensorflow-cpu` if GPU unavailable |

---

## 14. Future Work

1. **Hyperparameter Optimisation:** Use Optuna or Bayesian search over XGBoost and neural network hyperparameters

2. **SMOTE Oversampling:** Address class imbalance by oversampling minority attack classes, especially rare ones like Shellcode and Worms

3. **Graph Neural Network:** Model network traffic as a graph (nodes = IPs, edges = flows) for richer structural features

4. **SHAP Explainability:** Use SHAP values to explain individual predictions — critical for operational security teams

5. **Real-time Streaming Pipeline:** Wrap the trained model in a FastAPI service that accepts live network flows and returns predictions with latency < 100ms

6. **CIC-IDS-2017 Extension:** Add support for the larger CIC-IDS-2017 dataset to validate performance on a different traffic distribution

7. **Federated Learning:** Train across multiple network nodes without sharing raw traffic data (privacy-preserving)

---

## 15. References

1. Moustafa, N. & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems*. MilCIS 2015. IEEE.

2. Tavallaee, M., Bagheri, E., Lu, W. & Ghorbani, A.A. (2009). *A detailed analysis of the KDD CUP 99 data set*. IEEE CISDA 2009.

3. Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD 2016. ACM.

4. Hochreiter, S. & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735–1780.

5. Vincent, P. et al. (2010). *Stacked denoising autoencoders: Learning useful representations in a deep network*. JMLR, 11, 3371–3408.

6. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

7. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.

8. Scikit-learn: Machine Learning in Python (2011). Pedregosa et al. JMLR 12, 2825-2830.

---

*Document last updated: September 15, 2026*  
*Project recreated from 4th draft PDF: A2305220519_Deepanshu_Garhkoti_4nd_Draft.pdf*
