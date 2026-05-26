from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import OPTIMIZATION_CONFIG, OUTPUT_DIR, SIMULATION_CONFIG
from src.utils import get_logger, write_dataframe


LOGGER = get_logger(__name__)


def load_simulation_inputs(
    order_plan_path: Path | None = None,
    demand_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_plan = pd.read_csv(order_plan_path or OUTPUT_DIR / "recommended_order_plan.csv")
    demand = pd.read_csv(demand_path or OUTPUT_DIR / "forecast_results.csv")
    demand = demand[demand["target"] == "weekly_demand"].copy()
    demand["item_description"] = demand["entity_label"].str.split(r"\s\|\s").str[-1]
    demand_summary = demand.groupby("item_description", as_index=False)["forecast"].mean().rename(columns={"forecast": "forecast_demand"})
    return order_plan, demand_summary


def run_scenario(
    order_plan: pd.DataFrame,
    demand_df: pd.DataFrame,
    scenario_name: str,
    demand_multiplier: float = 1.0,
    delay_shock: bool = False,
    inflation_multiplier: float = 1.0,
    n_simulations: int = SIMULATION_CONFIG.n_simulations,
) -> pd.DataFrame:
    merged = order_plan.merge(demand_df, on="item_description", how="left")
    results: List[Dict[str, float | str | int]] = []

    for simulation_id in range(1, n_simulations + 1):
        service_levels = []
        stockout_costs = []
        total_cost = 0.0

        for row in merged.itertuples():
            demand = max(
                0,
                np.random.normal(
                    loc=(row.forecast_demand or 0) * demand_multiplier,
                    scale=max((row.forecast_demand or 0) * SIMULATION_CONFIG.demand_std_pct, 1),
                ),
            )
            supply = row.recommended_order_qty or 0
            if delay_shock and row.supplier_risk_score >= 66:
                delayed_units = np.random.binomial(int(max(round(supply), 0)), 0.15)
                supply = max(supply - delayed_units, 0)

            inflated_cost = row.estimated_total_cost * inflation_multiplier
            service_level = min(supply / demand, 1.0) if demand > 0 else 1.0
            stockout_units = max(demand - supply, 0)
            stockout_cost = stockout_units * (row.unit_price or 0) * OPTIMIZATION_CONFIG.stockout_penalty_rate

            service_levels.append(service_level)
            stockout_costs.append(stockout_cost)
            total_cost += inflated_cost + stockout_cost

        results.append(
            {
                "scenario": scenario_name,
                "simulation_id": simulation_id,
                "avg_service_level": float(np.mean(service_levels)) if service_levels else 1.0,
                "total_stockout_cost": float(np.sum(stockout_costs)),
                "total_cost": float(total_cost),
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    order_plan, demand_df = load_simulation_inputs()
    scenarios = [
        {"scenario_name": "Base case", "demand_multiplier": 1.0, "delay_shock": False, "inflation_multiplier": 1.0},
        {"scenario_name": "High demand case", "demand_multiplier": 1.20, "delay_shock": False, "inflation_multiplier": 1.0},
        {"scenario_name": "Supplier delay case", "demand_multiplier": 1.0, "delay_shock": True, "inflation_multiplier": 1.0},
        {"scenario_name": "Cost inflation case", "demand_multiplier": 1.0, "delay_shock": False, "inflation_multiplier": 1.08},
    ]

    scenario_frames = [run_scenario(order_plan, demand_df, **scenario) for scenario in scenarios]
    simulation_results = pd.concat(scenario_frames, ignore_index=True)

    service_level_distribution = simulation_results[["scenario", "simulation_id", "avg_service_level"]]
    expected_stockout_cost = (
        simulation_results.groupby("scenario", as_index=False)["total_stockout_cost"]
        .mean()
        .rename(columns={"total_stockout_cost": "expected_stockout_cost"})
    )
    scenario_summary = (
        simulation_results.groupby("scenario", as_index=False)
        .agg(
            avg_service_level=("avg_service_level", "mean"),
            p10_service_level=("avg_service_level", lambda x: np.percentile(x, 10)),
            p90_service_level=("avg_service_level", lambda x: np.percentile(x, 90)),
            avg_total_cost=("total_cost", "mean"),
        )
    )

    write_dataframe(service_level_distribution, OUTPUT_DIR / "service_level_distribution.csv")
    write_dataframe(expected_stockout_cost, OUTPUT_DIR / "expected_stockout_cost.csv")
    write_dataframe(scenario_summary, OUTPUT_DIR / "scenario_summary.csv")
    LOGGER.info("Saved simulation outputs to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

