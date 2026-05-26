from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import APP_CONFIG, PROCESSED_DIR
from src.utils import get_logger, snake_case, write_dataframe


LOGGER = get_logger(__name__)

DATE_COLUMNS = [
    "po_sent_to_vendor_date",
    "scheduled_delivery_date",
    "delivered_to_client_date",
    "delivery_recorded_date",
]

NUMERIC_COLUMN_ALIASES: Dict[str, List[str]] = {
    "line_item_quantity": ["line_item_quantity"],
    "line_item_value": ["line_item_value"],
    "pack_price": ["pack_price"],
    "unit_price": ["unit_price"],
    "weight_kg": ["weight_kilograms", "weight_kg"],
    "freight_cost_usd": ["freight_cost_usd"],
    "line_item_insurance_usd": ["line_item_insurance_usd"],
}

SPECIAL_TEXT_VALUES = {
    "Freight Included in Commodity Cost": np.nan,
    "Invoiced Separately": np.nan,
    "Weight Captured Separately": np.nan,
    "Date Not Captured": np.nan,
    "N/A - From RDC": np.nan,
}


def load_base_file(path: Path | None = None) -> pd.DataFrame:
    base_path = path or PROCESSED_DIR / APP_CONFIG.clean_base_filename
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base file not found at {base_path}. Run src/data_ingestion.py first."
        )
    return pd.read_csv(base_path, low_memory=False)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: snake_case(column) for column in df.columns}
    return df.rename(columns=renamed)


def replace_special_text_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(SPECIAL_TEXT_VALUES)
    see_dn_mask = df.astype(str).apply(lambda col: col.str.contains("See DN", case=False, na=False))
    if see_dn_mask.any().any():
        LOGGER.info("Replacing 'See DN...' style values with missing values.")
        df = df.mask(see_dn_mask, np.nan)
    return df


def create_missing_value_flags(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[f"{column}_missing_flag"] = df[column].isna().astype(int)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[f"{column}_missing_flag"] = df[column].isna().astype(int)
            df[column] = pd.to_datetime(df[column], errors="coerce")
            LOGGER.info("Parsed date column %s with %s nulls.", column, int(df[column].isna().sum()))
    return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for target_column, aliases in NUMERIC_COLUMN_ALIASES.items():
        source_column = next((alias for alias in aliases if alias in df.columns), None)
        if not source_column:
            LOGGER.warning("Expected numeric column missing: %s", target_column)
            continue

        if source_column != target_column:
            df[target_column] = df[source_column]

        df[f"{target_column}_missing_flag"] = df[target_column].isna().astype(int)
        df[target_column] = (
            df[target_column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        LOGGER.info(
            "Converted numeric column %s with %s nulls.",
            target_column,
            int(df[target_column].isna().sum()),
        )
    return df


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        column
        for column in NUMERIC_COLUMN_ALIASES.keys()
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    for column in numeric_columns:
        median_value = df[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        df[column] = df[column].fillna(median_value)
        LOGGER.info("Imputed %s with median %.4f.", column, float(median_value) if pd.notna(median_value) else np.nan)

    text_columns = [column for column in df.columns if df[column].dtype == "object"]
    for column in text_columns:
        mode_values = df[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)

    return df


def summarize_cleaning(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows_before",
                "rows_after",
                "columns_before",
                "columns_after",
                "row_drop_pct",
            ],
            "value": [
                len(df_before),
                len(df_after),
                df_before.shape[1],
                df_after.shape[1],
                round((1 - len(df_after) / max(len(df_before), 1)) * 100, 2),
            ],
        }
    )
    return summary


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    original_df = df.copy()
    df = standardize_columns(df)
    df = replace_special_text_values(df)
    df = create_missing_value_flags(df, [snake_case(column) for column in original_df.columns])
    df = parse_dates(df)
    df = coerce_numeric_columns(df)
    df = impute_values(df)

    if "weight_kilograms" in df.columns and "weight_kg" not in df.columns:
        df["weight_kg"] = df["weight_kilograms"]

    LOGGER.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


def main() -> None:
    raw_df = load_base_file()
    processed_df = preprocess_dataset(raw_df)

    cleaned_path = PROCESSED_DIR / "cleaned_supply_chain.csv"
    quality_path = PROCESSED_DIR / "cleaning_summary.csv"

    write_dataframe(processed_df, cleaned_path)
    write_dataframe(summarize_cleaning(raw_df, processed_df), quality_path)
    LOGGER.info("Saved cleaned dataset to %s", cleaned_path)
    LOGGER.info("Saved cleaning summary to %s", quality_path)


if __name__ == "__main__":
    main()
