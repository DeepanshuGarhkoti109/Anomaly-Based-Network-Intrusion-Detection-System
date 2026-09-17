"""
ModelTrainer
============
High-level wrapper that trains, evaluates, and saves all classical ML and
deep-learning models in one place.

Intended for scripted / pipeline use.  The notebooks use the same underlying
classes but call them directly for more visibility.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate_model
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and persists all models for the NIDS project.

    Parameters
    ----------
    models_dir : str | None
        Directory for saved model files.  Defaults to ``<project_root>/models``.
    results_dir : str | None
        Directory for CSV result files.  Defaults to ``<project_root>/results``.
    random_state : int
        Global random seed.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        random_state: int = 42,
    ):
        root = Path(__file__).resolve().parent.parent.parent
        self.models_dir  = Path(models_dir)  if models_dir  else root / "models"
        self.results_dir = Path(results_dir) if results_dir else root / "results"
        self.random_state = random_state

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._trained: Dict = {}  # name → (model, metrics_dict)

    # ------------------------------------------------------------------
    # Individual training methods
    # ------------------------------------------------------------------

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Tuple:
        params = dict(C=1.0, solver="lbfgs", max_iter=1000,
                      random_state=self.random_state)
        params.update(kwargs)

        model = LogisticRegression(**params)
        with timer("Logistic Regression training"):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)

        self._trained["Logistic Regression"] = (model, metrics)
        joblib.dump(model, self.models_dir / "logistic_regression.pkl")
        logger.info("Logistic Regression saved.")
        return model, metrics

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Tuple:
        params = dict(n_estimators=100, max_depth=10, n_jobs=-1,
                      random_state=self.random_state)
        params.update(kwargs)

        model = RandomForestClassifier(**params)
        with timer("Random Forest training"):
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)

        self._trained["Random Forest"] = (model, metrics)
        joblib.dump(model, self.models_dir / "random_forest.pkl")
        logger.info("Random Forest saved.")
        return model, metrics

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ) -> Tuple:
        params = dict(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
        params.update(kwargs)

        model = XGBClassifier(**params)
        with timer("XGBoost training"):
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)

        self._trained["XGBoost"] = (model, metrics)
        joblib.dump(model, self.models_dir / "xgboost.pkl")
        logger.info("XGBoost saved.")
        return model, metrics

    # ------------------------------------------------------------------
    # Train all classical models
    # ------------------------------------------------------------------

    def train_all_classical(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """
        Train Logistic Regression, Random Forest, and XGBoost sequentially.

        Returns
        -------
        pd.DataFrame – comparison table indexed by model name.
        """
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(      X_train, y_train, X_test, y_test)
        self.train_xgboost(            X_train, y_train, X_test, y_test)

        return self.comparison_table()

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(self) -> pd.DataFrame:
        """Return a DataFrame comparing all trained models."""
        rows = []
        for name, (_, metrics) in self._trained.items():
            rows.append({
                "Model":     name,
                "Accuracy":  metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall":    metrics["recall"],
                "F1":        metrics["f1"],
                "ROC-AUC":   metrics["roc_auc"],
            })

        df = pd.DataFrame(rows).set_index("Model").round(4)
        df.to_csv(self.results_dir / "model_comparison.csv")
        logger.info("Model comparison saved to %s", self.results_dir / "model_comparison.csv")
        return df

    # ------------------------------------------------------------------
    # Best model selection
    # ------------------------------------------------------------------

    def best_model(self, metric: str = "f1"):
        """
        Return ``(name, model)`` for the model with the highest *metric* score.

        Parameters
        ----------
        metric : str
            One of ``'accuracy'``, ``'f1'``, ``'roc_auc'``, etc.
        """
        if not self._trained:
            raise RuntimeError("No models trained yet.")

        best_name = max(
            self._trained,
            key=lambda n: self._trained[n][1].get(metric, 0),
        )
        best_model_obj = self._trained[best_name][0]
        logger.info("Best model by %s: %s", metric, best_name)
        return best_name, best_model_obj
