"""
Preprocessor
============
Handles missing-value imputation, outlier clipping, categorical encoding,
and exposes a full sklearn-compatible pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Stateful preprocessing helper.

    Call the individual steps in order, or use ``fit_transform_pipeline`` for
    a single-call full pipeline.

    Parameters
    ----------
    scaling_method : str
        ``'standard'``, ``'minmax'``, or ``'robust'``.
    impute_strategy : str
        Strategy passed to ``sklearn.impute.SimpleImputer`` (``'median'``,
        ``'mean'``, or ``'most_frequent'``).
    """

    SCALERS = {
        "standard": StandardScaler,
        "minmax":   MinMaxScaler,
        "robust":   RobustScaler,
    }

    def __init__(
        self,
        scaling_method: str = "standard",
        impute_strategy: str = "median",
    ):
        if scaling_method not in self.SCALERS:
            raise ValueError(f"scaling_method must be one of {list(self.SCALERS)}")

        self.scaling_method = scaling_method
        self.impute_strategy = impute_strategy

        self._numeric_imputer: Optional[SimpleImputer] = None
        self._cat_imputer:     Optional[SimpleImputer] = None
        self._scaler          = None
        self._cat_mappings:    Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Step 1 – Missing values
    # ------------------------------------------------------------------

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_fill: str = "Unknown",
    ) -> pd.DataFrame:
        """
        Impute missing values in-place on a copy of *df*.

        Numeric columns → median (or *numeric_strategy*).
        Categorical / object columns → fill with *categorical_fill*.
        """
        df = df.copy()

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Numeric imputation
        if num_cols:
            self._numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df[num_cols] = self._numeric_imputer.fit_transform(df[num_cols])

        # Categorical imputation
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna(categorical_fill)

        missing_after = df.isnull().sum().sum()
        logger.info("Missing values after imputation: %d", missing_after)
        return df

    # ------------------------------------------------------------------
    # Step 2 – Outlier clipping
    # ------------------------------------------------------------------

    def clip_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lower_q: float = 0.01,
        upper_q: float = 0.99,
    ) -> pd.DataFrame:
        """
        Clip each column in *columns* to [lower_q, upper_q] quantile range.
        Safe for both int and float columns.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found – skipping clip.", col)
                continue
            lo = df[col].quantile(lower_q)
            hi = df[col].quantile(upper_q)
            df[col] = df[col].clip(lower=lo, upper=hi)
        logger.info("Outlier clipping applied to %d columns.", len(columns))
        return df

    # ------------------------------------------------------------------
    # Step 3 – Categorical encoding
    # ------------------------------------------------------------------

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "onehot",
        drop_first: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical columns.

        Parameters
        ----------
        method : ``'onehot'`` or ``'ordinal'``

        Returns
        -------
        (encoded_df, mappings_dict)
            *mappings_dict* maps column name → category-to-int dict (ordinal)
            or the list of new dummy column names (onehot).
        """
        df = df.copy()
        mappings: Dict = {}

        for col in columns:
            if col not in df.columns:
                logger.warning("Column '%s' not found – skipping encode.", col)
                continue

            if method == "ordinal":
                categories = df[col].astype(str).unique().tolist()
                cat_map = {cat: idx for idx, cat in enumerate(sorted(categories))}
                df[col] = df[col].astype(str).map(cat_map)
                mappings[col] = cat_map
                logger.debug("Ordinal encoded '%s' → %d categories", col, len(cat_map))

            else:  # onehot
                dummies = pd.get_dummies(
                    df[col].astype(str),
                    prefix=col,
                    drop_first=drop_first,
                )
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                mappings[col] = dummies.columns.tolist()
                logger.debug("One-hot encoded '%s' → %d dummies", col, len(dummies.columns))

        self._cat_mappings = mappings
        return df, mappings

    # ------------------------------------------------------------------
    # Step 4 – Feature scaling
    # ------------------------------------------------------------------

    def fit_scaler(self, X: pd.DataFrame) -> "Preprocessor":
        """Fit the scaler on *X* (training set only)."""
        ScalerClass = self.SCALERS[self.scaling_method]
        self._scaler = ScalerClass()
        self._scaler.fit(X)
        logger.info("%s scaler fitted on %d features.", self.scaling_method, X.shape[1])
        return self

    def transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted scaler to *X*."""
        if self._scaler is None:
            raise RuntimeError("Call fit_scaler() before transform_scaler().")
        return self._scaler.transform(X)

    def fit_transform_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one call (use only on training data)."""
        return self.fit_scaler(X).transform_scaler(X)

    # ------------------------------------------------------------------
    # Full pipeline convenience method
    # ------------------------------------------------------------------

    def fit_transform_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        drop_cols: Optional[List[str]] = None,
        outlier_lower_q: float = 0.01,
        outlier_upper_q: float = 0.99,
    ) -> Tuple[np.ndarray, pd.Series, List[str]]:
        """
        Run all preprocessing steps on *df* and return
        ``(X_scaled_array, y_series, feature_names)``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe from DataLoader.
        target_col : str
            Name of the binary label column.
        drop_cols : list[str] | None
            Additional columns to drop before processing (e.g. 'attack_cat').
        outlier_lower_q / outlier_upper_q : float
            Quantile bounds for outlier clipping.
        """
        drop_cols = drop_cols or []
        cols_to_drop = [c for c in drop_cols + [target_col] if c in df.columns]

        y = df[target_col].copy() if target_col in df.columns else None
        df_work = df.drop(columns=cols_to_drop, errors="ignore").copy()

        # 1. Missing values
        df_work = self.handle_missing_values(df_work)

        # 2. Encode categoricals
        cat_cols = df_work.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df_work, _ = self.encode_categoricals(df_work, cat_cols, method="onehot")

        # 3. Outlier clipping
        num_cols = df_work.select_dtypes(include=["int64", "float64"]).columns.tolist()
        df_work = self.clip_outliers(df_work, num_cols,
                                     lower_q=outlier_lower_q,
                                     upper_q=outlier_upper_q)

        # 4. Scale
        feature_names = df_work.columns.tolist()
        X_scaled = self.fit_transform_scaler(df_work)

        logger.info(
            "Pipeline complete. X shape=%s, y shape=%s",
            X_scaled.shape,
            y.shape if y is not None else "N/A",
        )
        return X_scaled, y, feature_names
