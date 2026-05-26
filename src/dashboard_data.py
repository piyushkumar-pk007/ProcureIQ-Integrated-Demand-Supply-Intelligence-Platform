from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import OUTPUT_DIR, PROCESSED_DIR


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def load_dashboard_datasets() -> Dict[str, pd.DataFrame]:
    datasets = {
        "engineered": _safe_read_csv(PROCESSED_DIR / "engineered_supply_chain.csv", parse_dates=["order_date", "week_start"]),
        "forecast_results": _safe_read_csv(OUTPUT_DIR / "forecast_results.csv", parse_dates=["period"]),
        "model_comparison": _safe_read_csv(OUTPUT_DIR / "model_comparison.csv"),
        "supplier_risk": _safe_read_csv(OUTPUT_DIR / "supplier_risk_scores.csv"),
        "optimization_summary": _safe_read_csv(OUTPUT_DIR / "optimization_summary.csv"),
        "cost_savings": _safe_read_csv(OUTPUT_DIR / "cost_savings_vs_baseline.csv"),
        "recommended_order_plan": _safe_read_csv(OUTPUT_DIR / "recommended_order_plan.csv"),
        "scenario_summary": _safe_read_csv(OUTPUT_DIR / "scenario_summary.csv"),
        "service_level_distribution": _safe_read_csv(OUTPUT_DIR / "service_level_distribution.csv"),
    }
    return datasets


def executive_metrics(datasets: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    engineered = datasets["engineered"]
    forecast_results = datasets["forecast_results"]
    cost_savings = datasets["cost_savings"]

    total_shipments = float(len(engineered)) if not engineered.empty else 0.0
    total_landed_cost = float(engineered.get("total_landed_cost", pd.Series(dtype=float)).sum()) if not engineered.empty else 0.0
    late_delivery_rate = float(engineered.get("is_late_delivery", pd.Series(dtype=float)).mean() * 100) if not engineered.empty else 0.0

    if not forecast_results.empty:
        mae = (forecast_results["actual"] - forecast_results["forecast"]).abs().mean()
        forecast_accuracy = max(0.0, 100 - float(mae))
    else:
        forecast_accuracy = 0.0

    if not cost_savings.empty:
        cheapest_cost = cost_savings.loc[cost_savings["strategy"] == "cheapest_supplier_only", "total_cost"]
        optimized_cost = cost_savings.loc[cost_savings["strategy"] == "optimized", "total_cost"]
        potential_savings = float((cheapest_cost.iloc[0] - optimized_cost.iloc[0])) if not cheapest_cost.empty and not optimized_cost.empty else 0.0
    else:
        potential_savings = 0.0

    return {
        "total_shipments": total_shipments,
        "total_landed_cost": total_landed_cost,
        "late_delivery_rate": late_delivery_rate,
        "forecast_accuracy": forecast_accuracy,
        "potential_savings": potential_savings,
    }

