import pandas as pd

from src.feature_engineering import engineer_features


def test_engineer_features_creates_delay_and_cost_columns():
    df = pd.DataFrame(
        {
            "vendor": ["A", "A"],
            "product_group": ["Medicines", "Medicines"],
            "item_description": ["Item 1", "Item 1"],
            "po_sent_to_vendor_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "scheduled_delivery_date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            "delivered_to_client_date": pd.to_datetime(["2024-01-07", "2024-01-10"]),
            "delivery_recorded_date": pd.to_datetime(["2024-01-08", "2024-01-11"]),
            "line_item_quantity": [10, 20],
            "line_item_value": [100, 220],
            "unit_price": [10, 11],
            "weight_kg": [5, 10],
            "freight_cost_usd": [20, 25],
            "line_item_insurance_usd": [2, 3],
        }
    )
    featured = engineer_features(df)
    assert "delivery_delay_days" in featured.columns
    assert "total_landed_cost" in featured.columns
    assert featured["is_late_delivery"].sum() == 1
    assert featured["item_weekly_demand"].notna().all()

