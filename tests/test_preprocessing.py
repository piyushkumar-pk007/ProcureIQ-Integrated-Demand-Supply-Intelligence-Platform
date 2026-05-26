import pandas as pd

from src.preprocessing import preprocess_dataset, standardize_columns


def test_standardize_columns_to_snake_case():
    df = pd.DataFrame(columns=["PO Sent to Vendor Date", "Weight (Kilograms)"])
    standardized = standardize_columns(df)
    assert "po_sent_to_vendor_date" in standardized.columns
    assert "weight_kilograms" in standardized.columns


def test_preprocess_handles_special_values_and_flags():
    df = pd.DataFrame(
        {
            "PO Sent to Vendor Date": ["2024-01-01"],
            "Scheduled Delivery Date": ["2024-01-10"],
            "Delivered to Client Date": ["2024-01-12"],
            "Delivery Recorded Date": ["Date Not Captured"],
            "Line Item Quantity": ["10"],
            "Line Item Value": ["100"],
            "Pack Price": ["5"],
            "Unit Price": ["10"],
            "Weight (Kilograms)": ["Weight Captured Separately"],
            "Freight Cost (USD)": ["Freight Included in Commodity Cost"],
            "Line Item Insurance (USD)": ["Invoiced Separately"],
        }
    )
    processed = preprocess_dataset(df)
    assert "delivery_recorded_date_missing_flag" in processed.columns
    assert processed["weight_kg"].notna().all()
    assert processed["freight_cost_usd"].notna().all()

