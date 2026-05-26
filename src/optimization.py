from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

from src.config import OPTIMIZATION_CONFIG, OUTPUT_DIR, PROCESSED_DIR, OptimizationConfig
from src.utils import get_logger, weighted_average, write_dataframe


LOGGER = get_logger(__name__)


def load_inputs(
    engineered_path: Path | None = None,
    forecast_path: Path | None = None,
    risk_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engineered = pd.read_csv(
        engineered_path or PROCESSED_DIR / "engineered_supply_chain.csv",
        parse_dates=["order_date", "week_start"],
    )
    forecast = pd.read_csv(forecast_path or OUTPUT_DIR / "forecast_results.csv", parse_dates=["period"])
    risk = pd.read_csv(risk_path or OUTPUT_DIR / "supplier_risk_scores.csv")
    return engineered, forecast, risk


def classify_abc(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("item_description", dropna=False)["total_landed_cost"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    total = summary["total_landed_cost"].sum()
    summary["cumulative_share"] = summary["total_landed_cost"].cumsum() / max(total, 1)
    summary["abc_class"] = np.select(
        [summary["cumulative_share"] <= 0.8, summary["cumulative_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    return summary[["item_description", "abc_class"]]


def build_forecast_demand(forecast_df: pd.DataFrame) -> pd.DataFrame:
    demand_forecast = (
        forecast_df[forecast_df["target"] == "weekly_demand"]
        .groupby("entity_label", as_index=False)
        .agg(forecast_demand=("forecast", "mean"))
    )
    demand_forecast["item_description"] = demand_forecast["entity_label"].str.split(r"\s\|\s").str[-1]
    return demand_forecast


def derive_supplier_item_parameters(engineered_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    params = (
        engineered_df.groupby(["vendor", "item_description"], dropna=False)
        .agg(
            unit_price=("unit_price", "mean"),
            line_item_quantity=("line_item_quantity", "sum"),
            freight_cost_usd=("freight_cost_usd", "mean"),
            freight_per_unit=("freight_cost_usd", lambda x: float(np.mean(x))),
            lead_time_days=("lead_time_days", "mean"),
            total_landed_cost=("total_landed_cost", "mean"),
        )
        .reset_index()
    )
    params = params.merge(
        risk_df[["vendor", "supplier_risk_score", "supplier_risk_band"]],
        on="vendor",
        how="left",
    )
    params["minimum_order_quantity"] = np.maximum(OPTIMIZATION_CONFIG.minimum_order_quantity, params["line_item_quantity"] * 0.05)
    return params


def build_historical_allocation(engineered_df: pd.DataFrame) -> pd.DataFrame:
    allocation = (
        engineered_df.groupby(["item_description", "vendor"], dropna=False)["line_item_quantity"]
        .sum()
        .reset_index()
    )
    allocation["allocation_share"] = allocation["line_item_quantity"] / allocation.groupby("item_description")["line_item_quantity"].transform("sum")
    return allocation[["item_description", "vendor", "allocation_share"]]


def optimize_procurement(
    demand_df: pd.DataFrame,
    supplier_item_df: pd.DataFrame,
    abc_df: pd.DataFrame,
    config: OptimizationConfig = OPTIMIZATION_CONFIG,
) -> pd.DataFrame:
    demand = demand_df.merge(abc_df, on="item_description", how="left")
    demand["abc_class"] = demand["abc_class"].fillna("C")
    demand["service_level"] = demand["abc_class"].map(config.abc_service_levels).fillna(config.service_level_target)
    demand["required_quantity"] = demand["forecast_demand"] * demand["service_level"]

    feasible = supplier_item_df.merge(demand[["item_description", "required_quantity"]], on="item_description", how="inner")
    feasible = feasible.dropna(subset=["vendor", "item_description"]).copy()
    if feasible.empty:
        raise ValueError("No feasible vendor-item pairs found for optimization.")

    problem = pulp.LpProblem("ProcureIQ_Optimization", pulp.LpMinimize)
    quantity_vars: Dict[Tuple[str, str], pulp.LpVariable] = {}
    order_vars: Dict[Tuple[str, str], pulp.LpVariable] = {}

    for row in feasible.itertuples():
        key = (row.vendor, row.item_description)
        quantity_vars[key] = pulp.LpVariable(f"x_{abs(hash(key))}", lowBound=0, cat="Continuous")
        order_vars[key] = pulp.LpVariable(f"y_{abs(hash(key))}", cat="Binary")

    problem += pulp.lpSum(
        quantity_vars[(row.vendor, row.item_description)] * (
            row.unit_price
            + row.freight_per_unit * config.freight_cost_weight
            + row.unit_price * config.holding_cost_rate
            + row.unit_price * (row.supplier_risk_score / 100.0) * config.risk_penalty_weight
        )
        for row in feasible.itertuples()
    )

    for item_row in demand.itertuples():
        item_pairs = [pair for pair in quantity_vars if pair[1] == item_row.item_description]
        problem += (
            pulp.lpSum(quantity_vars[pair] for pair in item_pairs) >= item_row.required_quantity,
            f"demand_{abs(hash(item_row.item_description))}",
        )

    risky_limit = config.max_supplier_allocation
    for item_row in demand.itertuples():
        item_pairs = [pair for pair in quantity_vars if pair[1] == item_row.item_description]
        risky_pairs = [
            pair
            for pair in item_pairs
            if feasible.loc[
                (feasible["vendor"] == pair[0]) & (feasible["item_description"] == pair[1]),
                "supplier_risk_score",
            ].iloc[0]
            >= 66
        ]
        if risky_pairs:
            problem += (
                pulp.lpSum(quantity_vars[pair] for pair in risky_pairs) <= item_row.required_quantity * risky_limit,
                f"risky_alloc_{abs(hash(item_row.item_description))}",
            )

    problem += (
        pulp.lpSum(quantity_vars.values()) <= config.storage_capacity,
        "storage_capacity",
    )

    for row in feasible.itertuples():
        key = (row.vendor, row.item_description)
        problem += quantity_vars[key] <= config.big_m_quantity * order_vars[key]
        problem += quantity_vars[key] >= row.minimum_order_quantity * order_vars[key]

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)

    results = []
    for row in feasible.itertuples():
        key = (row.vendor, row.item_description)
        ordered_qty = quantity_vars[key].value() or 0.0
        if ordered_qty <= 0:
            continue
        results.append(
            {
                "vendor": row.vendor,
                "item_description": row.item_description,
                "recommended_order_qty": ordered_qty,
                "unit_price": row.unit_price,
                "freight_per_unit": row.freight_per_unit,
                "supplier_risk_score": row.supplier_risk_score,
                "estimated_total_cost": ordered_qty
                * (
                    row.unit_price
                    + row.freight_per_unit
                    + row.unit_price * config.holding_cost_rate
                    + row.unit_price * (row.supplier_risk_score / 100.0) * config.risk_penalty_weight
                ),
            }
        )

    return pd.DataFrame(results)


def baseline_cheapest_supplier(demand_df: pd.DataFrame, supplier_item_df: pd.DataFrame) -> pd.DataFrame:
    cheapest = supplier_item_df.sort_values(["item_description", "unit_price", "supplier_risk_score"]).groupby("item_description").first().reset_index()
    baseline = demand_df.merge(cheapest, on="item_description", how="left")
    baseline["recommended_order_qty"] = baseline["forecast_demand"]
    baseline["estimated_total_cost"] = baseline["recommended_order_qty"] * (
        baseline["unit_price"] + baseline["freight_per_unit"]
    )
    baseline["strategy"] = "cheapest_supplier_only"
    return baseline[["vendor", "item_description", "recommended_order_qty", "estimated_total_cost", "strategy"]]


def baseline_historical_allocation(
    demand_df: pd.DataFrame,
    supplier_item_df: pd.DataFrame,
    historical_allocation_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = demand_df.merge(historical_allocation_df, on="item_description", how="left").merge(
        supplier_item_df[["vendor", "item_description", "unit_price", "freight_per_unit"]],
        on=["vendor", "item_description"],
        how="left",
    )
    merged["recommended_order_qty"] = merged["forecast_demand"] * merged["allocation_share"].fillna(0)
    merged["estimated_total_cost"] = merged["recommended_order_qty"] * (
        merged["unit_price"].fillna(0) + merged["freight_per_unit"].fillna(0)
    )
    merged["strategy"] = "historical_vendor_allocation"
    return merged[["vendor", "item_description", "recommended_order_qty", "estimated_total_cost", "strategy"]]


def summarize_costs(
    optimized_df: pd.DataFrame,
    cheapest_df: pd.DataFrame,
    historical_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        {
            "strategy": ["optimized", "cheapest_supplier_only", "historical_vendor_allocation"],
            "total_cost": [
                optimized_df["estimated_total_cost"].sum(),
                cheapest_df["estimated_total_cost"].sum(),
                historical_df["estimated_total_cost"].sum(),
            ],
        }
    )
    optimized_cost = summary.loc[summary["strategy"] == "optimized", "total_cost"].iloc[0]
    comparison = summary.copy()
    comparison["savings_vs_optimized"] = comparison["total_cost"] - optimized_cost
    return summary, comparison


def save_allocation_plot(order_plan: pd.DataFrame) -> None:
    plot_df = order_plan.groupby("vendor", as_index=False)["recommended_order_qty"].sum().sort_values(
        "recommended_order_qty", ascending=False
    )
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["vendor"].head(12), plot_df["recommended_order_qty"].head(12))
    plt.title("Recommended Supplier Allocation")
    plt.xlabel("Recommended Quantity")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "supplier_allocation_chart.png", dpi=160)
    plt.close()


def main() -> None:
    engineered_df, forecast_df, risk_df = load_inputs()
    demand_df = build_forecast_demand(forecast_df)
    abc_df = classify_abc(engineered_df)
    supplier_item_df = derive_supplier_item_parameters(engineered_df, risk_df)
    historical_allocation_df = build_historical_allocation(engineered_df)

    optimized_df = optimize_procurement(demand_df, supplier_item_df, abc_df)
    optimized_df["strategy"] = "optimized"
    cheapest_df = baseline_cheapest_supplier(demand_df, supplier_item_df)
    historical_df = baseline_historical_allocation(demand_df, supplier_item_df, historical_allocation_df)

    summary_df, comparison_df = summarize_costs(optimized_df, cheapest_df, historical_df)

    write_dataframe(optimized_df, OUTPUT_DIR / "recommended_order_plan.csv")
    write_dataframe(summary_df, OUTPUT_DIR / "optimization_summary.csv")
    write_dataframe(comparison_df, OUTPUT_DIR / "cost_savings_vs_baseline.csv")
    save_allocation_plot(optimized_df)
    LOGGER.info("Saved optimization outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

