"""
pipeline.py
===========
End-to-end convenience function that wires together all project modules.

Usage
-----
>>> from src.pipeline import run_pipeline
>>> results = run_pipeline(dataset='unsw_nb15', model='all')
"""

import logging
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.utils.helpers import ensure_dir, set_random_seed

logger = logging.getLogger(__name__)

DATASET_TYPE  = Literal["unsw_nb15", "nsl_kdd"]
MODEL_TYPE    = Literal["all", "xgboost", "random_forest", "logistic_regression"]


def run_pipeline(
    dataset: DATASET_TYPE = "unsw_nb15",
    model: MODEL_TYPE = "all",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    feature_engineering: bool = True,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Execute the full NIDS ML pipeline.

    Parameters
    ----------
    dataset : str
        ``'unsw_nb15'`` or ``'nsl_kdd'``.
    model : str
        Which model(s) to train.  Use ``'all'`` to train all classical models.
    test_size : float
        Fraction of data held out for the test set.
    val_size : float
        Fraction of training data used for validation (not used by classical
        models but reserved for logging).
    random_state : int
        Global random seed.
    feature_engineering : bool
        Whether to apply the FeatureEngineer before training.
    output_dir : str | None
        Where to save models and results.  Defaults to project root.

    Returns
    -------
    pd.DataFrame – model comparison table.
    """
    set_random_seed(random_state)

    # Determine output directories
    project_root = Path(__file__).resolve().parent.parent
    output_dir   = Path(output_dir) if output_dir else project_root
    models_dir   = ensure_dir(str(output_dir / "models"))
    results_dir  = ensure_dir(str(output_dir / "final_reports"))

    logger.info("=== Pipeline START  dataset=%s  model=%s ===", dataset, model)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    loader = DataLoader(dataset=dataset)
    df     = loader.load_raw_data()
    logger.info("Data loaded: %s", df.shape)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    prep   = Preprocessor()
    X, y, feature_names = prep.fit_transform_pipeline(
        df,
        target_col="label",
        drop_cols=["attack_cat"],
    )
    logger.info("Preprocessing complete: X=%s  y=%s", X.shape, y.shape)

    # ------------------------------------------------------------------
    # 3. Feature engineering (optional)
    # ------------------------------------------------------------------
    if feature_engineering:
        X_df = pd.DataFrame(X, columns=feature_names)
        fe   = FeatureEngineer()
        X_df = fe.run_all(X_df)
        # Re-scale new features
        X    = prep.fit_transform_scaler(X_df)
        feature_names = X_df.columns.tolist()
        logger.info("Feature engineering done.  New shape: %s", X.shape)

    # ------------------------------------------------------------------
    # 4. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info("Train=%d  Test=%d", len(X_train), len(X_test))

    # ------------------------------------------------------------------
    # 5. Train model(s)
    # ------------------------------------------------------------------
    trainer = ModelTrainer(models_dir=models_dir, results_dir=results_dir,
                           random_state=random_state)

    y_train_arr = np.asarray(y_train)
    y_test_arr  = np.asarray(y_test)

    if model == "all":
        comparison = trainer.train_all_classical(X_train, y_train_arr,
                                                  X_test, y_test_arr)
    elif model == "xgboost":
        trainer.train_xgboost(X_train, y_train_arr, X_test, y_test_arr)
        comparison = trainer.comparison_table()
    elif model == "random_forest":
        trainer.train_random_forest(X_train, y_train_arr, X_test, y_test_arr)
        comparison = trainer.comparison_table()
    elif model == "logistic_regression":
        trainer.train_logistic_regression(X_train, y_train_arr, X_test, y_test_arr)
        comparison = trainer.comparison_table()
    else:
        raise ValueError(f"Unknown model '{model}'")

    logger.info("=== Pipeline DONE ===")
    return comparison
