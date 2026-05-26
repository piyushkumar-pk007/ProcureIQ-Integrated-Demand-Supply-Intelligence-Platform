from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.config import OUTPUT_DIR, PROCESSED_DIR
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)

try:
    import shap
except Exception:
    shap = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def load_engineered_dataset(path: Path | None = None) -> pd.DataFrame:
    engineered_path = path or PROCESSED_DIR / "engineered_supply_chain.csv"
    if not engineered_path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {engineered_path}. Run src/feature_engineering.py first."
        )
    return pd.read_csv(engineered_path, parse_dates=["order_date", "week_start"])


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def build_rule_based_risk(df: pd.DataFrame) -> pd.DataFrame:
    missing_flag_cols = [column for column in df.columns if column.endswith("_missing_flag")]
    vendor_metrics = (
        df.groupby("vendor", dropna=False)
        .agg(
            on_time_delivery_rate=("is_late_delivery", lambda x: 1 - np.mean(x)),
            average_delay_days=("delivery_delay_days", "mean"),
            delay_volatility=("delivery_delay_days", "std"),
            price_volatility_cv=("unit_price", lambda x: np.std(x) / np.mean(x) if np.mean(x) else 0),
            shipment_volume=("line_item_quantity", "sum"),
            freight_cost_volatility=("freight_cost_usd", "std"),
            missing_data_rate=(missing_flag_cols[0], "mean") if missing_flag_cols else ("is_late_delivery", "mean"),
        )
        .reset_index()
    )

    country_conc = (
        df.groupby(["vendor", "country"]).size().reset_index(name="shipments")
        if "country" in df.columns
        else pd.DataFrame(columns=["vendor", "country", "shipments"])
    )
    if not country_conc.empty:
        country_conc["share"] = country_conc["shipments"] / country_conc.groupby("vendor")["shipments"].transform("sum")
        country_risk = country_conc.groupby("vendor")["share"].max().reset_index(name="country_concentration_risk")
        vendor_metrics = vendor_metrics.merge(country_risk, on="vendor", how="left")
    else:
        vendor_metrics["country_concentration_risk"] = 0.0

    product_conc = (
        df.groupby(["vendor", "product_group"]).size().reset_index(name="shipments")
        if "product_group" in df.columns
        else pd.DataFrame(columns=["vendor", "product_group", "shipments"])
    )
    if not product_conc.empty:
        product_conc["share"] = product_conc["shipments"] / product_conc.groupby("vendor")["shipments"].transform("sum")
        product_risk = product_conc.groupby("vendor")["share"].max().reset_index(name="product_concentration_risk")
        vendor_metrics = vendor_metrics.merge(product_risk, on="vendor", how="left")
    else:
        vendor_metrics["product_concentration_risk"] = 0.0

    vendor_metrics = vendor_metrics.fillna(0)
    vendor_metrics["late_rate"] = 1 - vendor_metrics["on_time_delivery_rate"]

    score_components = {
        "late_rate": 0.25,
        "average_delay_days": 0.15,
        "delay_volatility": 0.10,
        "price_volatility_cv": 0.15,
        "freight_cost_volatility": 0.10,
        "missing_data_rate": 0.10,
        "country_concentration_risk": 0.075,
        "product_concentration_risk": 0.075,
    }

    vendor_metrics["supplier_risk_score"] = 0.0
    for component, weight in score_components.items():
        vendor_metrics["supplier_risk_score"] += _normalize(vendor_metrics[component]) * weight * 100

    vendor_metrics["supplier_risk_score"] = vendor_metrics["supplier_risk_score"].clip(0, 100)
    vendor_metrics["supplier_risk_band"] = pd.cut(
        vendor_metrics["supplier_risk_score"],
        bins=[-0.1, 33, 66, 100],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return vendor_metrics


def train_late_delivery_model(df: pd.DataFrame) -> Dict[str, object]:
    feature_columns = [
        "line_item_quantity",
        "unit_price",
        "freight_cost_usd",
        "line_item_insurance_usd",
        "weight_kg",
        "lead_time_days",
        "vendor_volume_share",
        "vendor_price_cv",
        "vendor_avg_delay",
        "vendor_late_rate",
    ]
    available_features = [column for column in feature_columns if column in df.columns]
    modeling_df = df.dropna(subset=["is_late_delivery", "order_date"]).sort_values("order_date").copy()
    modeling_df = modeling_df[available_features + ["is_late_delivery"]].replace([np.inf, -np.inf], np.nan)

    if len(modeling_df) < 50 or modeling_df["is_late_delivery"].nunique() < 2:
        LOGGER.warning("Not enough data to train ML supplier risk model.")
        return {"model_type": "insufficient_data"}

    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        model_type = "xgboost_classifier"
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
        model_type = "random_forest_classifier"

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )

    splitter = TimeSeriesSplit(n_splits=3)
    auc_scores = []
    for train_idx, test_idx in splitter.split(modeling_df):
        x_train = modeling_df.iloc[train_idx][available_features]
        y_train = modeling_df.iloc[train_idx]["is_late_delivery"]
        x_test = modeling_df.iloc[test_idx][available_features]
        y_test = modeling_df.iloc[test_idx]["is_late_delivery"]
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        auc_scores.append(roc_auc_score(y_test, probabilities))

    pipeline.fit(modeling_df[available_features], modeling_df["is_late_delivery"])

    return {
        "model_type": model_type,
        "features": available_features,
        "pipeline": pipeline,
        "mean_auc": float(np.mean(auc_scores)),
    }


def save_feature_importance_plot(model_artifacts: Dict[str, object], filename: str) -> None:
    if "pipeline" not in model_artifacts:
        return

    model = model_artifacts["pipeline"].named_steps["model"]
    features = model_artifacts["features"]
    if not hasattr(model, "feature_importances_"):
        return

    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    feature_importances.plot(kind="barh")
    plt.title("Supplier Late-Delivery Model Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def save_shap_plot(df: pd.DataFrame, model_artifacts: Dict[str, object], filename: str) -> None:
    if shap is None or "pipeline" not in model_artifacts:
        return
    model = model_artifacts["pipeline"].named_steps["model"]
    features = model_artifacts["features"]
    if not hasattr(model, "feature_importances_"):
        return

    sample = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median()).head(200)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def save_risk_plots(risk_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    risk_df["supplier_risk_score"].plot(kind="hist", bins=15)
    plt.title("Supplier Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supplier_risk_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_df = risk_df.sort_values("supplier_risk_score", ascending=False).head(10)
    plt.barh(plot_df["vendor"], plot_df["supplier_risk_score"])
    plt.title("Top Risky Suppliers")
    plt.xlabel("Risk Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_risky_suppliers.png", dpi=160)
    plt.close()


def main() -> None:
    df = load_engineered_dataset()
    risk_df = build_rule_based_risk(df)
    model_artifacts = train_late_delivery_model(df)

    if "pipeline" in model_artifacts:
        feature_frame = df.groupby("vendor", as_index=False)[model_artifacts["features"]].mean(numeric_only=True)
        probabilities = model_artifacts["pipeline"].predict_proba(feature_frame[model_artifacts["features"]])[:, 1]
        probability_df = pd.DataFrame(
            {
                "vendor": feature_frame["vendor"],
                "ml_late_delivery_probability": probabilities,
            }
        )
        risk_df = risk_df.merge(probability_df, on="vendor", how="left")
        risk_df["ml_model_type"] = model_artifacts["model_type"]
        risk_df["ml_mean_auc"] = model_artifacts["mean_auc"]
        save_feature_importance_plot(model_artifacts, "supplier_risk_feature_importance.png")
        save_shap_plot(df, model_artifacts, "supplier_risk_shap_summary.png")
    else:
        risk_df["ml_late_delivery_probability"] = np.nan
        risk_df["ml_model_type"] = model_artifacts.get("model_type", "not_available")
        risk_df["ml_mean_auc"] = np.nan

    top_risky = risk_df.sort_values("supplier_risk_score", ascending=False).head(15)

    write_dataframe(risk_df, OUTPUT_DIR / "supplier_risk_scores.csv")
    write_dataframe(top_risky, OUTPUT_DIR / "top_risky_suppliers.csv")
    save_risk_plots(risk_df)
    LOGGER.info("Saved supplier risk outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
