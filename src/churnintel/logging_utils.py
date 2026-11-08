"""Consistent, quiet logging for scripts and the API."""

from __future__ import annotations

import logging
import sys
import warnings


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("matplotlib", "PIL", "shap", "numba", "urllib3", "httpx", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")
