"""
helpers.py
==========
General-purpose utility functions: seeding, timing, and memory reporting.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow (if installed)
    to ensure reproducible experiments.

    Parameters
    ----------
    seed : int
        Seed value.  42 used as the project default.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug("TensorFlow seed set to %d", seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        logger.debug("PyTorch seed set to %d", seed)
    except ImportError:
        pass

    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def timer(label: str = ""):
    """
    Context manager that prints elapsed time.

    Usage
    -----
    >>> with timer("XGBoost training"):
    ...     model.fit(X_train, y_train)
    XGBoost training: 2.34 s
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        msg = f"{label}: {elapsed:.2f} s" if label else f"Elapsed: {elapsed:.2f} s"
        print(msg)
        logger.info(msg)


def timeit(func: Callable) -> Callable:
    """
    Decorator that prints the execution time of *func*.

    Usage
    -----
    >>> @timeit
    ... def train_model():
    ...     ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        msg = f"{func.__name__} completed in {elapsed:.2f} s"
        print(msg)
        logger.info(msg)
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------


def memory_usage(obj=None) -> str:
    """
    Return a human-readable memory usage string.

    If *obj* is a pandas DataFrame, return its memory usage.
    Otherwise return the current process RSS.

    Parameters
    ----------
    obj : pd.DataFrame | None

    Returns
    -------
    str – e.g. ``"256.4 MB"``
    """
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            bytes_ = obj.memory_usage(deep=True).sum()
            return _fmt_bytes(bytes_)
    except ImportError:
        pass

    try:
        import psutil
        process = psutil.Process(os.getpid())
        bytes_ = process.memory_info().rss
        return _fmt_bytes(bytes_)
    except ImportError:
        return "psutil not installed"


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist.  Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path
