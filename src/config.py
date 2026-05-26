from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"
ASSETS_DIR = ROOT_DIR / "assets"


@dataclass
class OptimizationConfig:
    storage_capacity: float = 500_000.0
    minimum_order_quantity: float = 100.0
    service_level_target: float = 0.92
    holding_cost_rate: float = 0.08
    stockout_penalty_rate: float = 0.25
    max_supplier_allocation: float = 0.55
    risk_penalty_weight: float = 0.15
    freight_cost_weight: float = 1.0
    big_m_quantity: float = 1_000_000.0
    abc_service_levels: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.95, "B": 0.90, "C": 0.85}
    )


@dataclass
class SimulationConfig:
    n_simulations: int = 500
    demand_std_pct: float = 0.15
    inflation_mean_pct: float = 0.05
    inflation_std_pct: float = 0.02
    extra_delay_days_high_risk: int = 7


@dataclass
class ForecastConfig:
    min_history_periods: int = 18
    top_series_count: int = 5
    forecast_horizon: int = 8
    fallback_frequency: str = "M"


@dataclass
class AppConfig:
    project_name: str = "ProcureIQ — Integrated Demand-Supply Intelligence Platform"
    dataset_slug: str = "apoorvwatsky/supply-chain-shipment-pricing-data"
    clean_base_filename: str = "clean_base.csv"
    featured_metrics: Dict[str, str] = field(
        default_factory=lambda: {
            "demand_target": "weekly line_item_quantity",
            "price_target": "weekly weighted average unit_price",
            "risk_target": "supplier reliability and late-delivery risk",
        }
    )


OPTIMIZATION_CONFIG = OptimizationConfig()
SIMULATION_CONFIG = SimulationConfig()
FORECAST_CONFIG = ForecastConfig()
APP_CONFIG = AppConfig()

