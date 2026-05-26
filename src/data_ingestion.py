from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from src.config import APP_CONFIG, PROCESSED_DIR, RAW_DIR
from src.utils import detect_csv_file, ensure_directories, get_logger, write_dataframe


LOGGER = get_logger(__name__)


def _download_from_kaggle() -> Optional[Path]:
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    dataset_slug = os.getenv("KAGGLE_DATASET", APP_CONFIG.dataset_slug)

    if not username or not key:
        LOGGER.info("Kaggle credentials not found. Skipping Kaggle download.")
        return None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        LOGGER.warning("kaggle package is not installed. Falling back to manual CSV mode.")
        return None

    api = KaggleApi()
    api.authenticate()
    LOGGER.info("Downloading Kaggle dataset: %s", dataset_slug)
    api.dataset_download_files(dataset_slug, path=str(RAW_DIR), force=False, quiet=False)

    zip_files = sorted(RAW_DIR.glob("*.zip"))
    for archive_path in zip_files:
        LOGGER.info("Extracting archive: %s", archive_path.name)
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(RAW_DIR)
        archive_path.unlink(missing_ok=True)

    return detect_csv_file(RAW_DIR)


def locate_source_csv() -> Path:
    ensure_directories()
    csv_path = detect_csv_file(RAW_DIR)
    if csv_path:
        LOGGER.info("Using existing raw CSV: %s", csv_path)
        return csv_path

    csv_path = _download_from_kaggle()
    if csv_path:
        LOGGER.info("Using Kaggle-downloaded CSV: %s", csv_path)
        return csv_path

    raise FileNotFoundError(
        "No CSV file found in data/raw/. Download the Kaggle dataset manually and place the main CSV there."
    )


def save_clean_intermediate(csv_path: Path) -> pd.DataFrame:
    LOGGER.info("Loading source CSV from %s", csv_path)
    last_error: Exception | None = None
    for encoding in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding=encoding)
            LOGGER.info("Loaded CSV using encoding: %s", encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error if last_error else RuntimeError("Unable to read CSV with supported encodings.")

    output_path = PROCESSED_DIR / APP_CONFIG.clean_base_filename
    write_dataframe(df, output_path)
    LOGGER.info("Saved clean intermediate copy to %s", output_path)
    return df


def copy_raw_reference(csv_path: Path) -> None:
    target_path = PROCESSED_DIR / "raw_source_reference.csv"
    if not target_path.exists():
        shutil.copyfile(csv_path, target_path)


def main() -> None:
    csv_path = locate_source_csv()
    save_clean_intermediate(csv_path)
    copy_raw_reference(csv_path)


if __name__ == "__main__":
    main()
