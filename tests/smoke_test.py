"""
Smoke test – verifies that the core modules work end-to-end using the
built-in synthetic dataset (no real data download required).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.features.engineer import FeatureEngineer
from src.evaluation.metrics import evaluate_model
from src.utils.helpers import set_random_seed, timer


def test_loader():
    loader = DataLoader(dataset="unsw_nb15")
    df = loader.load_raw_data()
    assert df.shape[0] > 0, f"Expected non-empty dataframe, got {df.shape[0]}"
    assert "label" in df.columns, "'label' column missing"
    assert df["label"].isin([0, 1]).all(), "label must be binary (0/1)"
    print(f"  [loader]       PASS  shape={df.shape}")
    return df


def test_preprocessor(df):
    prep = Preprocessor()
    X, y, feats = prep.fit_transform_pipeline(
        df, target_col="label", drop_cols=["attack_cat"]
    )
    assert X.shape[0] == len(df), "Row count changed after preprocessing"
    assert not np.isnan(X).any(), "NaN values found after preprocessing"
    print(f"  [preprocessor] PASS  X={X.shape}  y={y.shape}")
    return X, y, feats


def test_feature_engineer(X, feats):
    X_df = pd.DataFrame(X, columns=feats)
    fe = FeatureEngineer()
    X_eng = fe.run_all(X_df)
    assert X_eng.shape[1] >= X_df.shape[1], "Feature count should not decrease"
    assert len(fe.new_feature_names) > 0, "No new features were created"
    print(f"  [engineer]     PASS  new_features={len(fe.new_feature_names)}  total={X_eng.shape[1]}")
    return X_eng


def test_evaluation(X, y):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    # Use a fast baseline: majority-class predictor
    majority = int(np.bincount(y_tr.astype(int)).argmax())
    y_pred = np.full(len(y_te), majority)
    y_prob = np.zeros(len(y_te), dtype=float)

    metrics = evaluate_model(y_te, y_pred, y_prob, verbose=False)
    required_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"
    print(f"  [evaluation]   PASS  accuracy={metrics['accuracy']:.4f}")


def test_xgboost_train(X, y):
    from xgboost import XGBClassifier

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    model = XGBClassifier(
        n_estimators=20, max_depth=3,
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = evaluate_model(y_te, y_pred, y_prob, verbose=False)
    assert metrics["f1"] > 0.5, f"XGBoost F1 too low: {metrics['f1']:.4f}"
    print(f"  [xgboost]      PASS  F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    set_random_seed(42)
    print("Running smoke tests...\n")

    with timer("Total"):
        df = test_loader()
        X, y, feats = test_preprocessor(df)
        X_eng = test_feature_engineer(X, feats)
        test_evaluation(X, y)
        test_xgboost_train(X, y)

    print("\nAll smoke tests PASSED.")
