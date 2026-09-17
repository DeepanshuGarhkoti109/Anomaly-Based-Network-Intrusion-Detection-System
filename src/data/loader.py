"""
DataLoader
==========
Loads raw UNSW-NB15 and NSL-KDD CSV files from the data/ directory.

UNSW-NB15 column reference
---------------------------
49 features + 2 label columns (label = binary, attack_cat = multi-class)

NSL-KDD column reference
-------------------------
41 features, no header row – column names supplied in KDD_COLUMNS.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UNSW-NB15 column metadata
# ---------------------------------------------------------------------------
UNSW_FEATURE_GROUPS = {
    "basic":       ["srcip", "sport", "dstip", "dsport", "proto"],
    "content":     ["state", "dur", "sbytes", "dbytes", "sttl", "dttl",
                    "sloss", "dloss", "service", "sload", "dload",
                    "spkts", "dpkts"],
    "time":        ["swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
                    "trans_depth", "res_bdy_len", "sjit", "djit",
                    "stime", "ltime", "sintpkt", "dintpkt"],
    "connection":  ["tcprtt", "synack", "ackdat", "is_sm_ips_ports",
                    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
                    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                    "ct_dst_sport_ltm", "ct_dst_src_ltm"],
    "labels":      ["attack_cat", "label"],
}

UNSW_ATTACK_CATEGORIES = [
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]

# ---------------------------------------------------------------------------
# NSL-KDD column names (41 features + 2 label columns)
# ---------------------------------------------------------------------------
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty_level",
]

KDD_ATTACK_MAP = {
    "normal":      "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}


class DataLoader:
    """
    Unified loader for UNSW-NB15 and NSL-KDD datasets.

    Parameters
    ----------
    dataset : str
        Either ``'unsw_nb15'`` or ``'nsl_kdd'``.
    data_dir : str | None
        Path to the data directory.  Defaults to ``<project_root>/data``.
    """

    SUPPORTED = {"unsw_nb15", "nsl_kdd"}

    def __init__(self, dataset: str = "unsw_nb15", data_dir: str | None = None):
        if dataset not in self.SUPPORTED:
            raise ValueError(f"dataset must be one of {self.SUPPORTED}, got '{dataset}'")

        self.dataset = dataset

        # Resolve project root: go two levels up from this file (src/data/loader.py)
        src_data_dir = Path(__file__).resolve().parent
        project_root = src_data_dir.parent.parent

        self.data_dir = Path(data_dir) if data_dir else project_root / "data"
        self.raw_dir = self.data_dir / "raw"

        logger.info("DataLoader initialised  dataset=%s  data_dir=%s",
                    self.dataset, self.data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw_data(self) -> pd.DataFrame:
        """Return the raw (unprocessed) dataset as a DataFrame."""
        if self.dataset == "unsw_nb15":
            return self._load_unsw_nb15()
        return self._load_nsl_kdd()

    def load_preprocessed_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Return (X_train, X_test, y_train, y_test) from the processed/ directory.
        Raises FileNotFoundError if preprocessing has not been run yet.
        """
        proc_dir = self.data_dir / "processed"
        required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        for fname in required:
            if not (proc_dir / fname).exists():
                raise FileNotFoundError(
                    f"{proc_dir / fname} not found.  "
                    "Run Notebook 02_preprocessing.ipynb first."
                )

        X_train = pd.read_csv(proc_dir / "X_train.csv")
        X_test  = pd.read_csv(proc_dir / "X_test.csv")
        y_train = pd.read_csv(proc_dir / "y_train.csv").squeeze()
        y_test  = pd.read_csv(proc_dir / "y_test.csv").squeeze()

        logger.info("Loaded preprocessed splits from %s", proc_dir)
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # UNSW-NB15 helpers
    # ------------------------------------------------------------------

    def _load_unsw_nb15(self) -> pd.DataFrame:
        """
        Try multiple common filenames / locations.
        Falls back to generating a synthetic demo dataset when no file found.
        """
        candidates = [
            self.raw_dir / "UNSW_NB15_training-set.csv",
            self.raw_dir / "UNSW-NB15.csv",
            self.data_dir / "UNSW_NB15_training-set.csv",
            self.data_dir / "UNSW-NB15.csv",
        ]

        for path in candidates:
            if path.exists():
                logger.info("Loading UNSW-NB15 from %s", path)
                df = pd.read_csv(path, low_memory=False)
                df = self._clean_unsw_nb15(df)
                return df

        logger.warning(
            "UNSW-NB15 file not found in any of %s.  "
            "Generating synthetic demo dataset (10 000 rows).  "
            "Download the real data from https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            [str(p) for p in candidates],
        )
        return self._synthetic_unsw_nb15(n_samples=10_000)

    def _clean_unsw_nb15(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names and ensure label columns exist."""
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Some versions ship 'label' as 'labels'
        if "labels" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"labels": "label"})

        # Ensure binary label is integer
        if "label" in df.columns:
            df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

        return df

    def _synthetic_unsw_nb15(self, n_samples: int = 10_000) -> pd.DataFrame:
        """
        Generate a realistic synthetic UNSW-NB15-like dataset for development /
        demonstration purposes when the real files are not available.
        """
        rng = np.random.default_rng(42)

        n_normal = int(n_samples * 0.57)
        n_attack = n_samples - n_normal

        def _normal_block(n):
            return {
                "dur":       rng.exponential(0.1, n),
                "sbytes":    rng.integers(40, 1500, n),
                "dbytes":    rng.integers(40, 1500, n),
                "sttl":      rng.choice([64, 128, 255], n),
                "dttl":      rng.choice([64, 128, 255], n),
                "sloss":     rng.integers(0, 2, n),
                "dloss":     rng.integers(0, 2, n),
                "sload":     rng.uniform(0, 1e4, n),
                "dload":     rng.uniform(0, 1e4, n),
                "spkts":     rng.integers(1, 20, n),
                "dpkts":     rng.integers(1, 20, n),
                "swin":      rng.choice([255, 512, 1024, 8192, 65535], n),
                "dwin":      rng.choice([255, 512, 1024, 8192, 65535], n),
                "sjit":      rng.uniform(0, 10, n),
                "djit":      rng.uniform(0, 10, n),
                "ct_srv_src": rng.integers(1, 30, n),
                "ct_srv_dst": rng.integers(1, 30, n),
                "ct_dst_ltm": rng.integers(1, 10, n),
                "ct_src_ltm": rng.integers(1, 10, n),
                "proto":     rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.2, 0.1]),
                "service":   rng.choice(["http", "dns", "ftp", "-", "smtp"], n),
                "state":     rng.choice(["FIN", "CON", "INT", "REQ"], n),
                "label":     np.zeros(n, dtype=int),
                "attack_cat": np.full(n, "Normal"),
            }

        def _attack_block(n):
            block = _normal_block(n)
            attack_types = rng.choice(
                UNSW_ATTACK_CATEGORIES[1:], n,
                p=[0.05, 0.05, 0.30, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02],
            )
            block["sbytes"]    = rng.integers(0, 50000, n)
            block["dbytes"]    = rng.integers(0, 50000, n)
            block["sloss"]     = rng.integers(0, 100, n)
            block["spkts"]     = rng.integers(1, 5000, n)
            block["sjit"]      = rng.uniform(0, 500, n)
            block["label"]     = np.ones(n, dtype=int)
            block["attack_cat"] = attack_types
            return block

        normal = pd.DataFrame(_normal_block(n_normal))
        attack = pd.DataFrame(_attack_block(n_attack))

        df = pd.concat([normal, attack], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info("Synthetic UNSW-NB15 dataset created  shape=%s", df.shape)
        return df

    # ------------------------------------------------------------------
    # NSL-KDD helpers
    # ------------------------------------------------------------------

    def _load_nsl_kdd(self) -> pd.DataFrame:
        candidates = [
            self.raw_dir / "KDDTrain+.txt",
            self.raw_dir / "KDDTrain+.csv",
            self.data_dir / "KDDTrain+.txt",
        ]

        for path in candidates:
            if path.exists():
                logger.info("Loading NSL-KDD from %s", path)
                df = pd.read_csv(path, header=None, names=KDD_COLUMNS, low_memory=False)
                df = self._clean_nsl_kdd(df)
                return df

        logger.warning(
            "NSL-KDD file not found.  "
            "Generating synthetic demo dataset (10 000 rows).  "
            "Download from https://www.unb.ca/cic/datasets/nsl.html"
        )
        return self._synthetic_nsl_kdd(n_samples=10_000)

    def _clean_nsl_kdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw attack labels to binary 0/1 and broad category."""
        df["attack_type"] = df["attack_type"].str.strip().str.lower().str.rstrip(".")
        df["attack_cat"] = df["attack_type"].map(KDD_ATTACK_MAP).fillna("Other")
        df["label"] = (df["attack_type"] != "normal").astype(int)
        return df

    def _synthetic_nsl_kdd(self, n_samples: int = 10_000) -> pd.DataFrame:
        """Minimal synthetic NSL-KDD-like dataset for demo purposes."""
        rng = np.random.default_rng(42)
        n_normal = int(n_samples * 0.53)
        n_attack = n_samples - n_normal

        rows = []
        for i in range(n_samples):
            is_attack = i >= n_normal
            rows.append({
                "duration":       rng.integers(0, 60000),
                "protocol_type":  rng.choice(["tcp", "udp", "icmp"]),
                "service":        rng.choice(["http", "ftp", "smtp", "other"]),
                "flag":           rng.choice(["SF", "S0", "REJ", "RSTO"]),
                "src_bytes":      rng.integers(0, 100000 if is_attack else 10000),
                "dst_bytes":      rng.integers(0, 100000 if is_attack else 10000),
                "land":           rng.integers(0, 2),
                "wrong_fragment": rng.integers(0, 5 if is_attack else 2),
                "urgent":         rng.integers(0, 2),
                "hot":            rng.integers(0, 30),
                "num_failed_logins": rng.integers(0, 6 if is_attack else 2),
                "logged_in":      rng.integers(0, 2),
                "num_compromised": rng.integers(0, 10 if is_attack else 2),
                "root_shell":     rng.integers(0, 2),
                "su_attempted":   rng.integers(0, 2),
                "num_root":       rng.integers(0, 10 if is_attack else 2),
                "num_file_creations": rng.integers(0, 10),
                "num_shells":     rng.integers(0, 5),
                "num_access_files": rng.integers(0, 10),
                "num_outbound_cmds": 0,
                "is_host_login":  rng.integers(0, 2),
                "is_guest_login": rng.integers(0, 2),
                "count":          rng.integers(1, 512),
                "srv_count":      rng.integers(1, 512),
                "serror_rate":    rng.uniform(0, 1 if is_attack else 0.1),
                "srv_serror_rate": rng.uniform(0, 1),
                "rerror_rate":    rng.uniform(0, 1),
                "srv_rerror_rate": rng.uniform(0, 1),
                "same_srv_rate":  rng.uniform(0, 1),
                "diff_srv_rate":  rng.uniform(0, 1),
                "srv_diff_host_rate": rng.uniform(0, 1),
                "dst_host_count": rng.integers(1, 256),
                "dst_host_srv_count": rng.integers(1, 256),
                "dst_host_same_srv_rate": rng.uniform(0, 1),
                "dst_host_diff_srv_rate": rng.uniform(0, 1),
                "dst_host_same_src_port_rate": rng.uniform(0, 1),
                "dst_host_srv_diff_host_rate": rng.uniform(0, 1),
                "dst_host_serror_rate": rng.uniform(0, 1),
                "dst_host_srv_serror_rate": rng.uniform(0, 1),
                "dst_host_rerror_rate": rng.uniform(0, 1),
                "dst_host_srv_rerror_rate": rng.uniform(0, 1),
                "attack_type":    rng.choice(["neptune", "smurf", "ipsweep"]) if is_attack else "normal",
                "difficulty_level": rng.integers(1, 22),
                "label":          int(is_attack),
                "attack_cat":     rng.choice(["DoS", "Probe"]) if is_attack else "Normal",
            })

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Synthetic NSL-KDD dataset created  shape=%s", df.shape)
        return df
