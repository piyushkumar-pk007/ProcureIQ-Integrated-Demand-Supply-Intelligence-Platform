from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(OUTPUT_DIR / "pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def snake_case(text: str) -> str:
    text = text.strip().replace("%", "pct").replace("/", " ").replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.lower().strip("_")


def detect_csv_file(directory: Path) -> Optional[Path]:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        return None
    preferred = [path for path in csv_files if "supply" in path.name.lower()]
    return preferred[0] if preferred else csv_files[0]


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        denominator_series = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
        denominator_series = denominator_series.replace({0: np.nan})
        return numerator / denominator_series
    return float(numerator) / float(denominator) if denominator not in (0, 0.0) else np.nan


def weighted_average(values: Iterable[float], weights: Iterable[float]) -> float:
    values_arr = np.asarray(list(values), dtype=float)
    weights_arr = np.asarray(list(weights), dtype=float)
    weight_sum = weights_arr.sum()
    if weight_sum == 0 or len(values_arr) == 0:
        return float(np.nan)
    return float(np.average(values_arr, weights=weights_arr))


def write_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def optional_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None

