from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.utils import get_logger, safe_divide, weighted_average, write_dataframe


LOGGER = get_logger(__name__)


def _get_series(df: pd.DataFrame, column: str, default: float | str = 0) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_cleaned_dataset(path: Path | None = None) -> pd.DataFrame:
    cleaned_path = path or PROCESSED_DIR / "cleaned_supply_chain.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_path}. Run src/preprocessing.py first."
        )
    df = pd.read_csv(cleaned_path, parse_dates=[
        "po_sent_to_vendor_date",
        "scheduled_delivery_date",
        "delivered_to_client_date",
        "delivery_recorded_date",
    ])
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    order_date = df.get("po_sent_to_vendor_date")
    if order_date is None:
        fallback_candidates = [
            "scheduled_delivery_date",
            "delivered_to_client_date",
            "delivery_recorded_date",
        ]
        for candidate in fallback_candidates:
            if candidate in df.columns:
                order_date = df[candidate]
                break
    df["order_date"] = pd.to_datetime(order_date, errors="coerce")
    df["order_month"] = df["order_date"].dt.month
    df["order_week"] = df["order_date"].dt.isocalendar().week.astype("Int64")
    df["order_year"] = df["order_date"].dt.year
    return df


def add_cost_and_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delivery_delay_days"] = (
        pd.to_datetime(df["delivered_to_client_date"], errors="coerce")
        - pd.to_datetime(df["scheduled_delivery_date"], errors="coerce")
    ).dt.days
    df["is_late_delivery"] = (df["delivery_delay_days"] > 0).astype(int)
    df["verification_delay_days"] = (
        pd.to_datetime(df["delivery_recorded_date"], errors="coerce")
        - pd.to_datetime(df["delivered_to_client_date"], errors="coerce")
    ).dt.days
    df["total_landed_cost"] = (
        _get_series(df, "line_item_value").fillna(0)
        + _get_series(df, "freight_cost_usd").fillna(0)
        + _get_series(df, "line_item_insurance_usd").fillna(0)
    )
    df["freight_per_kg"] = safe_divide(_get_series(df, "freight_cost_usd"), _get_series(df, "weight_kg"))
    df["value_per_unit"] = safe_divide(_get_series(df, "line_item_value"), _get_series(df, "line_item_quantity"))
    df["weighted_unit_price"] = safe_divide(
        _get_series(df, "unit_price") * _get_series(df, "line_item_quantity"),
        _get_series(df, "line_item_quantity"),
    )
    df["lead_time_days"] = (
        pd.to_datetime(df["scheduled_delivery_date"], errors="coerce")
        - pd.to_datetime(df["po_sent_to_vendor_date"], errors="coerce")
    ).dt.days
    return df


def add_vendor_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "vendor" not in df.columns:
        df["vendor"] = "Unknown Vendor"

    total_qty = df["line_item_quantity"].sum()
    vendor_stats = df.groupby("vendor").agg(
        vendor_total_quantity=("line_item_quantity", "sum"),
        vendor_price_mean=("unit_price", "mean"),
        vendor_price_std=("unit_price", "std"),
        vendor_avg_delay=("delivery_delay_days", "mean"),
        vendor_delay_std=("delivery_delay_days", "std"),
        vendor_late_rate=("is_late_delivery", "mean"),
    )
    vendor_stats["vendor_volume_share"] = vendor_stats["vendor_total_quantity"] / max(total_qty, 1)
    vendor_stats["vendor_price_cv"] = (
        vendor_stats["vendor_price_std"] / vendor_stats["vendor_price_mean"].replace({0: np.nan})
    )
    df = df.merge(
        vendor_stats[
            [
                "vendor_volume_share",
                "vendor_price_cv",
                "vendor_avg_delay",
                "vendor_late_rate",
            ]
        ],
        on="vendor",
        how="left",
    )
    return df


def add_weekly_item_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "item_description" not in df.columns:
        df["item_description"] = _get_series(df, "product_group", "Unknown Item")

    df["week_start"] = df["order_date"].dt.to_period("W").dt.start_time
    weekly_metrics = (
        df.groupby(["item_description", "week_start"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "item_weekly_demand": group["line_item_quantity"].sum(),
                    "item_weekly_avg_price": weighted_average(
                        group["unit_price"].fillna(0),
                        group["line_item_quantity"].fillna(0),
                    ),
                }
            )
        )
        .reset_index()
    )
    df = df.merge(weekly_metrics, on=["item_description", "week_start"], how="left")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_temporal_features(df)
    df = add_cost_and_delay_features(df)
    df = add_vendor_aggregates(df)
    df = add_weekly_item_metrics(df)
    LOGGER.info("Feature engineering complete. Final shape: %s", df.shape)
    return df


def save_feature_outputs(df: pd.DataFrame) -> None:
    write_dataframe(df, PROCESSED_DIR / "engineered_supply_chain.csv")

    weekly_snapshot = (
        df.groupby(["week_start", "product_group"], dropna=False)
        .agg(
            weekly_quantity=("line_item_quantity", "sum"),
            weekly_value=("line_item_value", "sum"),
            weekly_weighted_price=("unit_price", "mean"),
        )
        .reset_index()
    )
    write_dataframe(weekly_snapshot, PROCESSED_DIR / "weekly_snapshot.csv")


def main() -> None:
    df = load_cleaned_dataset()
    engineered_df = engineer_features(df)
    save_feature_outputs(engineered_df)


if __name__ == "__main__":
    main()
