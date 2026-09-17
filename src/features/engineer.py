"""
FeatureEngineer
===============
Constructs domain-specific and statistical features on top of the
preprocessed UNSW-NB15 / NSL-KDD data.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates additional features from network traffic data.

    All public methods accept and return a DataFrame, so they can be
    chained:

    >>> fe = FeatureEngineer()
    >>> df_out = fe.add_ratio_features(df).add_log_features(df)
    """

    # Columns present in UNSW-NB15 that describe byte / packet volumes
    BYTE_COLS   = ["sbytes", "dbytes"]
    PACKET_COLS = ["spkts", "dpkts"]
    LOAD_COLS   = ["sload", "dload"]
    JITTER_COLS = ["sjit", "djit"]

    def __init__(self):
        self._new_features: List[str] = []

    # ------------------------------------------------------------------
    # Ratio features  (byte / packet ratios, etc.)
    # ------------------------------------------------------------------

    def add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add byte-ratio and packet-ratio features.

        New columns
        -----------
        bytes_ratio   : sbytes / (sbytes + dbytes + 1)
        packets_ratio : spkts  / (spkts  + dpkts  + 1)
        load_ratio    : sload  / (sload  + dload  + 1)
        bytes_per_pkt : (sbytes + dbytes) / (spkts + dpkts + 1)
        """
        df = df.copy()

        if all(c in df.columns for c in ["sbytes", "dbytes"]):
            total_bytes = df["sbytes"] + df["dbytes"] + 1
            df["bytes_ratio"]   = df["sbytes"] / total_bytes
            self._new_features.append("bytes_ratio")

        if all(c in df.columns for c in ["spkts", "dpkts"]):
            total_pkts = df["spkts"] + df["dpkts"] + 1
            df["packets_ratio"] = df["spkts"] / total_pkts
            self._new_features.append("packets_ratio")

        if all(c in df.columns for c in ["sload", "dload"]):
            total_load = df["sload"] + df["dload"] + 1
            df["load_ratio"]    = df["sload"] / total_load
            self._new_features.append("load_ratio")

        if all(c in df.columns for c in ["sbytes", "dbytes", "spkts", "dpkts"]):
            total_pkts = df["spkts"] + df["dpkts"] + 1
            df["bytes_per_pkt"] = (df["sbytes"] + df["dbytes"]) / total_pkts
            self._new_features.append("bytes_per_pkt")

        logger.debug("Ratio features added: %s", self._new_features)
        return df

    # ------------------------------------------------------------------
    # Log-transform features  (handles zero / negative safely)
    # ------------------------------------------------------------------

    def add_log_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply ``log1p`` to heavily right-skewed numeric columns.

        Parameters
        ----------
        columns : list[str] | None
            Columns to transform.  Defaults to byte and packet columns.
        """
        df = df.copy()
        columns = columns or (self.BYTE_COLS + self.PACKET_COLS + self.LOAD_COLS)

        added = []
        for col in columns:
            if col in df.columns:
                new_col = f"{col}_log"
                df[new_col] = np.log1p(np.clip(df[col], 0, None))
                added.append(new_col)

        self._new_features.extend(added)
        logger.debug("Log features added: %s", added)
        return df

    # ------------------------------------------------------------------
    # Statistical interaction features
    # ------------------------------------------------------------------

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add higher-order statistical features.

        New columns
        -----------
        jitter_sum   : sjit + djit
        jitter_ratio : sjit / (sjit + djit + 1e-9)
        ttl_diff     : |sttl - dttl|   (if present)
        win_diff     : |swin - dwin|   (if present)
        """
        df = df.copy()
        added = []

        if all(c in df.columns for c in ["sjit", "djit"]):
            df["jitter_sum"]   = df["sjit"] + df["djit"]
            df["jitter_ratio"] = df["sjit"] / (df["sjit"] + df["djit"] + 1e-9)
            added += ["jitter_sum", "jitter_ratio"]

        if all(c in df.columns for c in ["sttl", "dttl"]):
            df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()
            added.append("ttl_diff")

        if all(c in df.columns for c in ["swin", "dwin"]):
            df["win_diff"] = (df["swin"] - df["dwin"]).abs()
            added.append("win_diff")

        self._new_features.extend(added)
        logger.debug("Statistical features added: %s", added)
        return df

    # ------------------------------------------------------------------
    # Connection frequency features
    # ------------------------------------------------------------------

    def add_connection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive features based on connection-count columns present in
        UNSW-NB15 (ct_* columns).
        """
        df = df.copy()
        added = []

        ct_src = "ct_srv_src"
        ct_dst = "ct_srv_dst"

        if ct_src in df.columns and ct_dst in df.columns:
            df["ct_ratio"] = df[ct_src] / (df[ct_src] + df[ct_dst] + 1)
            added.append("ct_ratio")

        ct_src_ltm = "ct_src_ltm"
        ct_dst_ltm = "ct_dst_ltm"

        if ct_src_ltm in df.columns and ct_dst_ltm in df.columns:
            df["ct_ltm_ratio"] = df[ct_src_ltm] / (df[ct_src_ltm] + df[ct_dst_ltm] + 1)
            added.append("ct_ltm_ratio")

        self._new_features.extend(added)
        logger.debug("Connection features added: %s", added)
        return df

    # ------------------------------------------------------------------
    # Convenience: run all engineering steps
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps in sequence."""
        df = (
            self.add_ratio_features(df)
                .pipe(self.add_log_features)
                .pipe(self.add_statistical_features)
                .pipe(self.add_connection_features)
        )
        logger.info("Feature engineering complete.  New columns: %d  Total: %d",
                    len(self._new_features), df.shape[1])
        return df

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def new_feature_names(self) -> List[str]:
        """Names of all features added so far."""
        return list(self._new_features)
