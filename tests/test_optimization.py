import pandas as pd

from src.config import OptimizationConfig
from src.optimization import optimize_procurement


def test_optimize_procurement_meets_demand():
    demand_df = pd.DataFrame(
        {
            "item_description": ["Item 1"],
            "forecast_demand": [150.0],
        }
    )
    supplier_item_df = pd.DataFrame(
        {
            "vendor": ["Vendor A", "Vendor B"],
            "item_description": ["Item 1", "Item 1"],
            "unit_price": [10.0, 12.0],
            "line_item_quantity": [500.0, 500.0],
            "freight_cost_usd": [1.0, 1.5],
            "freight_per_unit": [1.0, 1.5],
            "lead_time_days": [10.0, 12.0],
            "total_landed_cost": [11.0, 13.5],
            "supplier_risk_score": [25.0, 70.0],
            "supplier_risk_band": ["Low Risk", "High Risk"],
            "minimum_order_quantity": [50.0, 50.0],
        }
    )
    abc_df = pd.DataFrame({"item_description": ["Item 1"], "abc_class": ["A"]})
    config = OptimizationConfig(storage_capacity=500.0, minimum_order_quantity=50.0, max_supplier_allocation=0.4)

    result = optimize_procurement(demand_df, supplier_item_df, abc_df, config=config)

    assert not result.empty
    assert result["recommended_order_qty"].sum() >= 150.0 * 0.95

