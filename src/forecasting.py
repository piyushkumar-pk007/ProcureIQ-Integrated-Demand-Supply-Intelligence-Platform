from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import FORECAST_CONFIG, OUTPUT_DIR, PROCESSED_DIR
from src.utils import get_logger, weighted_average, write_dataframe


LOGGER = get_logger(__name__)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None


@dataclass
class ForecastSeries:
    series_id: str
    entity_label: str
    target_name: str
    frequency: str
    data: pd.DataFrame


def load_engineered_dataset(path: Path | None = None) -> pd.DataFrame:
    engineered_path = path or PROCESSED_DIR / "engineered_supply_chain.csv"
    if not engineered_path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {engineered_path}. Run src/feature_engineering.py first."
        )
    return pd.read_csv(engineered_path, parse_dates=["order_date", "week_start"])


def _text_series(df: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column in df.columns:
        return df[column].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _weighted_price(group: pd.DataFrame) -> float:
    return weighted_average(group["unit_price"], group["line_item_quantity"])


def aggregate_time_series(
    df: pd.DataFrame,
    entity_columns: List[str],
    frequency: str = "W",
) -> pd.DataFrame:
    aggregated = (
        df.dropna(subset=["order_date"])
        .groupby(entity_columns + [pd.Grouper(key="order_date", freq=frequency)], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "weekly_demand": group["line_item_quantity"].sum(),
                    "weekly_weighted_avg_price": _weighted_price(group),
                }
            )
        )
        .reset_index()
        .rename(columns={"order_date": "period"})
    )
    return aggregated


def prepare_forecast_series(df: pd.DataFrame) -> List[ForecastSeries]:
    base_df = df.copy()
    base_df["item_key"] = (
        _text_series(base_df, "product_group", "Unknown")
        + " | "
        + _text_series(base_df, "item_description", "Unknown")
    )

    weekly = aggregate_time_series(base_df, ["product_group", "item_key"], "W")
    counts = (
        weekly.groupby(["product_group", "item_key"])
        .size()
        .reset_index(name="n_periods")
        .sort_values("n_periods", ascending=False)
    )

    sufficient = counts[counts["n_periods"] >= FORECAST_CONFIG.min_history_periods].head(
        FORECAST_CONFIG.top_series_count
    )

    frequency = "W"
    entity_columns = ["product_group", "item_key"]
    source = weekly

    if sufficient.empty:
        LOGGER.warning(
            "Not enough weekly depth found. Falling back to product_group + month aggregation."
        )
        source = aggregate_time_series(base_df, ["product_group"], FORECAST_CONFIG.fallback_frequency)
        counts = (
            source.groupby(["product_group"])
            .size()
            .reset_index(name="n_periods")
            .sort_values("n_periods", ascending=False)
        )
        sufficient = counts[counts["n_periods"] >= 8].head(FORECAST_CONFIG.top_series_count)
        frequency = FORECAST_CONFIG.fallback_frequency
        entity_columns = ["product_group"]

    series_collection: List[ForecastSeries] = []
    for _, row in sufficient.iterrows():
        mask = pd.Series(True, index=source.index)
        if "item_key" in sufficient.columns and "item_key" in source.columns:
            mask = mask & (source["item_key"] == row["item_key"])
        if "product_group" in source.columns:
            mask = mask & (source["product_group"] == row["product_group"])

        entity_df = source.loc[mask].sort_values("period").copy()
        label_parts = [str(row[col]) for col in entity_columns if col in row]
        label = " | ".join(label_parts)
        series_collection.append(
            ForecastSeries(
                series_id=label.replace(" | ", "__").replace("/", "_"),
                entity_label=label,
                target_name="weekly_demand",
                frequency=frequency,
                data=entity_df[["period", "weekly_demand", "weekly_weighted_avg_price"]].copy(),
            )
        )
    return series_collection


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom * 100) if denom else np.nan


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }


def seasonal_naive_forecast(train: pd.Series, horizon: int, seasonal_period: int = 4) -> np.ndarray:
    if len(train) < seasonal_period:
        last_value = train.iloc[-1]
        return np.repeat(last_value, horizon)
    repeated = np.resize(train.iloc[-seasonal_period:].to_numpy(), horizon)
    return repeated


def moving_average_forecast(train: pd.Series, horizon: int, window: int = 4) -> np.ndarray:
    avg = train.iloc[-window:].mean() if len(train) >= window else train.mean()
    return np.repeat(avg, horizon)


def sarima_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if SARIMAX is None or len(train) < 8:
        return moving_average_forecast(train, horizon)
    try:
        model = SARIMAX(
            train.astype(float),
            order=(1, 1, 1),
            seasonal_order=(1, 0, 0, 4),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        return fitted.forecast(horizon).to_numpy()
    except Exception as exc:
        LOGGER.warning("SARIMA failed: %s", exc)
        return moving_average_forecast(train, horizon)


def create_lag_features(series_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = series_df.copy()
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    for lag in [1, 2, 4, 8]:
        df[f"lag_{lag}"] = df[target_column].shift(lag)
    df["rolling_mean_4"] = df[target_column].shift(1).rolling(4).mean()
    df["rolling_std_4"] = df[target_column].shift(1).rolling(4).std()
    return df.dropna().reset_index(drop=True)


def ml_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> np.ndarray:
    model_df = pd.concat([train_df, test_df], ignore_index=True)
    model_df = create_lag_features(model_df[["period", target_column]].copy(), target_column)

    if model_df.empty:
        return moving_average_forecast(train_df[target_column], len(test_df))

    split_point = len(create_lag_features(train_df[["period", target_column]].copy(), target_column))
    train_features = model_df.iloc[:split_point].copy()
    test_features = model_df.iloc[split_point:].copy()
    if test_features.empty or train_features.empty:
        return moving_average_forecast(train_df[target_column], len(test_df))

    feature_columns = [column for column in train_features.columns if column not in {"period", target_column}]
    x_train = train_features[feature_columns]
    y_train = train_features[target_column]
    x_test = test_features[feature_columns]

    if XGBRegressor is not None:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=250, random_state=42)

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    if len(preds) < len(test_df):
        padding = np.repeat(preds[-1], len(test_df) - len(preds))
        preds = np.concatenate([preds, padding])
    return preds[: len(test_df)]


def prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> np.ndarray:
    if Prophet is None:
        return moving_average_forecast(train_df[target_column], len(test_df))
    prophet_train = train_df[["period", target_column]].rename(columns={"period": "ds", target_column: "y"})
    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_train)
    future = model.make_future_dataframe(periods=len(test_df), freq="W")
    forecast = model.predict(future)
    return forecast["yhat"].tail(len(test_df)).to_numpy()


def evaluate_models_for_series(series: ForecastSeries, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = series.data[["period", target_column]].dropna().reset_index(drop=True)
    if len(data) < 10:
        return pd.DataFrame(), pd.DataFrame()

    horizon = min(FORECAST_CONFIG.forecast_horizon, max(2, len(data) // 4))
    tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(data) // horizon - 1)))

    all_results = []
    model_functions: Dict[str, Callable] = {
        "seasonal_naive": lambda train_df, test_df: seasonal_naive_forecast(train_df[target_column], len(test_df)),
        "moving_average": lambda train_df, test_df: moving_average_forecast(train_df[target_column], len(test_df)),
        "sarima": lambda train_df, test_df: sarima_forecast(train_df[target_column], len(test_df)),
        "xgboost_or_rf": lambda train_df, test_df: ml_forecast(train_df, test_df, target_column),
        "prophet_optional": lambda train_df, test_df: prophet_forecast(train_df, test_df, target_column),
    }

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(data), start=1):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        for model_name, model_function in model_functions.items():
            try:
                preds = np.asarray(model_function(train_df, test_df), dtype=float)
                metrics = compute_metrics(test_df[target_column].to_numpy(dtype=float), preds)
                all_results.append(
                    {
                        "series_id": series.series_id,
                        "entity_label": series.entity_label,
                        "target": target_column,
                        "frequency": series.frequency,
                        "fold": fold_id,
                        "model_name": model_name,
                        **metrics,
                    }
                )
            except Exception as exc:
                LOGGER.warning(
                    "Model %s failed for %s on %s: %s",
                    model_name,
                    series.entity_label,
                    target_column,
                    exc,
                )

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_df = (
        results_df.groupby(["series_id", "entity_label", "target", "frequency", "model_name"], as_index=False)[
            ["mae", "rmse", "mape", "wape"]
        ]
        .mean()
        .sort_values(["series_id", "wape", "rmse"])
    )

    best_forecasts = []
    for _, row in summary_df.groupby(["series_id", "target"]).first().reset_index().iterrows():
        model_name = row["model_name"]
        final_train = data.iloc[:-horizon].copy()
        final_test = data.iloc[-horizon:].copy()
        preds = model_functions[model_name](final_train, final_test)
        forecast_frame = final_test.copy()
        forecast_frame["series_id"] = series.series_id
        forecast_frame["entity_label"] = series.entity_label
        forecast_frame["target"] = target_column
        forecast_frame["model_name"] = model_name
        forecast_frame["actual"] = final_test[target_column].to_numpy()
        forecast_frame["forecast"] = preds
        best_forecasts.append(forecast_frame[["series_id", "entity_label", "target", "period", "actual", "forecast", "model_name"]])

    return summary_df, pd.concat(best_forecasts, ignore_index=True)


def save_forecast_plot(forecast_df: pd.DataFrame, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df["period"], forecast_df["actual"], marker="o", label="Actual")
    plt.plot(forecast_df["period"], forecast_df["forecast"], marker="o", label="Forecast")
    plt.title(f"{forecast_df['entity_label'].iloc[0]} | {forecast_df['target'].iloc[0]}")
    plt.xlabel("Period")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def main() -> None:
    df = load_engineered_dataset()
    series_list = prepare_forecast_series(df)
    if not series_list:
        raise ValueError("No forecastable series were detected. Check order_date and quantity fields.")

    model_summaries = []
    forecasts = []
    for series in series_list:
        for target_column in ["weekly_demand", "weekly_weighted_avg_price"]:
            summary_df, forecast_df = evaluate_models_for_series(series, target_column)
            if not summary_df.empty:
                model_summaries.append(summary_df)
            if not forecast_df.empty:
                forecasts.append(forecast_df)
                safe_name = f"forecast_{series.series_id}_{target_column}.png"
                save_forecast_plot(forecast_df, safe_name)

    if not model_summaries or not forecasts:
        raise ValueError("Forecasting completed without usable results.")

    model_comparison = pd.concat(model_summaries, ignore_index=True)
    forecast_results = pd.concat(forecasts, ignore_index=True)

    write_dataframe(model_comparison, OUTPUT_DIR / "model_comparison.csv")
    write_dataframe(forecast_results, OUTPUT_DIR / "forecast_results.csv")
    LOGGER.info("Saved forecast outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
