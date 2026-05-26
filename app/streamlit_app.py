from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import APP_CONFIG, OptimizationConfig
from src.dashboard_data import executive_metrics, load_dashboard_datasets
from src.optimization import (
    baseline_cheapest_supplier,
    baseline_historical_allocation,
    build_forecast_demand,
    build_historical_allocation,
    classify_abc,
    derive_supplier_item_parameters,
    optimize_procurement,
    summarize_costs,
)
from src.simulation import run_scenario


st.set_page_config(page_title="ProcureIQ", layout="wide")


def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_missing_data_message() -> None:
    st.info(
        "Run the pipeline first: `python src/data_ingestion.py`, `python src/preprocessing.py`, "
        "`python src/feature_engineering.py`, `python src/forecasting.py`, `python src/supplier_risk.py`, "
        "`python src/optimization.py`, and `python src/simulation.py`."
    )


datasets = load_dashboard_datasets()
metrics = executive_metrics(datasets)

st.title(APP_CONFIG.project_name)
st.caption(
    "This is a Kaggle-based interview demonstration inspired by real supply-chain procurement and forecasting problems."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Forecasting",
        "Supplier Risk",
        "Procurement Optimizer",
        "Scenario Simulator",
    ],
)

if page == "Executive Overview":
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Shipments", f"{metrics['total_shipments']:,.0f}")
    col2.metric("Total Landed Cost", fmt_currency(metrics["total_landed_cost"]))
    col3.metric("Late Delivery Rate", f"{metrics['late_delivery_rate']:.1f}%")
    col4.metric("Forecast Accuracy", f"{metrics['forecast_accuracy']:.1f}")
    col5.metric("Potential Savings", fmt_currency(metrics["potential_savings"]))

    engineered = datasets["engineered"]
    if engineered.empty:
        render_missing_data_message()
    else:
        st.subheader("Shipment Volume by Product Group")
        if "product_group" in engineered.columns:
            chart_df = (
                engineered.groupby("product_group", as_index=False)["line_item_quantity"]
                .sum()
                .sort_values("line_item_quantity", ascending=False)
                .head(10)
            )
            st.plotly_chart(
                px.bar(chart_df, x="product_group", y="line_item_quantity", color="line_item_quantity"),
                use_container_width=True,
            )

        st.subheader("Delivery Delay Distribution")
        if "delivery_delay_days" in engineered.columns:
            st.plotly_chart(
                px.histogram(engineered, x="delivery_delay_days", nbins=40),
                use_container_width=True,
            )

if page == "Forecasting":
    forecast_results = datasets["forecast_results"]
    model_comparison = datasets["model_comparison"]
    if forecast_results.empty:
        render_missing_data_message()
    else:
        series_options = sorted(forecast_results["entity_label"].dropna().unique())
        selected_series = st.selectbox("Select product group / item", series_options)
        selected_target = st.selectbox("Select target", sorted(forecast_results["target"].dropna().unique()))

        filtered = forecast_results[
            (forecast_results["entity_label"] == selected_series)
            & (forecast_results["target"] == selected_target)
        ].sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered["period"], y=filtered["actual"], mode="lines+markers", name="Actual"))
        fig.add_trace(go.Scatter(x=filtered["period"], y=filtered["forecast"], mode="lines+markers", name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Comparison")
        comparison = model_comparison[model_comparison["entity_label"] == selected_series].sort_values("wape")
        st.dataframe(comparison, use_container_width=True)

if page == "Supplier Risk":
    risk_df = datasets["supplier_risk"]
    if risk_df.empty:
        render_missing_data_message()
    else:
        st.subheader("Supplier Risk Table")
        st.dataframe(
            risk_df.sort_values("supplier_risk_score", ascending=False),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                px.histogram(risk_df, x="supplier_risk_score", nbins=20, color="supplier_risk_band"),
                use_container_width=True,
            )
        with col2:
            top_risk = risk_df.sort_values("supplier_risk_score", ascending=False).head(10)
            st.plotly_chart(
                px.bar(top_risk, x="supplier_risk_score", y="vendor", orientation="h", color="supplier_risk_band"),
                use_container_width=True,
            )

        importance_note = "SHAP plot is saved to `data/outputs/` when the optional `shap` package is installed."
        st.caption(importance_note)

if page == "Procurement Optimizer":
    engineered = datasets["engineered"]
    forecast_results = datasets["forecast_results"]
    risk_df = datasets["supplier_risk"]
    if engineered.empty or forecast_results.empty or risk_df.empty:
        render_missing_data_message()
    else:
        st.sidebar.subheader("Optimizer Inputs")
        horizon = st.sidebar.slider("Forecast horizon", min_value=4, max_value=16, value=8)
        service_level = st.sidebar.slider("Default service level", min_value=0.80, max_value=0.99, value=0.92)
        risk_penalty = st.sidebar.slider("Risk penalty weight", min_value=0.00, max_value=1.00, value=0.15)
        storage_capacity = st.sidebar.number_input("Storage capacity", min_value=1000.0, value=500000.0, step=5000.0)

        tuned_config = OptimizationConfig(
            storage_capacity=storage_capacity,
            service_level_target=service_level,
            risk_penalty_weight=risk_penalty,
        )

        demand_df = build_forecast_demand(forecast_results.tail(len(forecast_results)))
        abc_df = classify_abc(engineered)
        supplier_item_df = derive_supplier_item_parameters(engineered, risk_df)
        historical_allocation_df = build_historical_allocation(engineered)

        optimized_df = optimize_procurement(demand_df, supplier_item_df, abc_df, config=tuned_config)
        cheapest_df = baseline_cheapest_supplier(demand_df, supplier_item_df)
        historical_df = baseline_historical_allocation(demand_df, supplier_item_df, historical_allocation_df)
        summary_df, comparison_df = summarize_costs(optimized_df, cheapest_df, historical_df)

        st.subheader("Recommended Order Plan")
        st.dataframe(optimized_df.sort_values("estimated_total_cost", ascending=False), use_container_width=True)

        st.subheader("Cost Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        st.plotly_chart(px.bar(summary_df, x="strategy", y="total_cost", color="strategy"), use_container_width=True)

if page == "Scenario Simulator":
    order_plan = datasets["recommended_order_plan"]
    forecast_results = datasets["forecast_results"]
    if order_plan.empty or forecast_results.empty:
        render_missing_data_message()
    else:
        demand_shock = st.toggle("Demand shock (+20%)", value=False)
        supplier_delay = st.toggle("Supplier delay shock", value=False)
        cost_inflation = st.toggle("Cost inflation (+8%)", value=False)

        demand_df = (
            forecast_results[forecast_results["target"] == "weekly_demand"]
            .assign(item_description=lambda frame: frame["entity_label"].str.split(r"\s\|\s").str[-1])
            .groupby("item_description", as_index=False)["forecast"]
            .mean()
            .rename(columns={"forecast": "forecast_demand"})
        )
        sim_df = run_scenario(
            order_plan,
            demand_df,
            scenario_name="Interactive",
            demand_multiplier=1.2 if demand_shock else 1.0,
            delay_shock=supplier_delay,
            inflation_multiplier=1.08 if cost_inflation else 1.0,
            n_simulations=500,
        )

        st.subheader("Simulation Results")
        col1, col2 = st.columns(2)
        col1.metric("Average Service Level", f"{sim_df['avg_service_level'].mean():.2%}")
        col2.metric("Average Total Cost", fmt_currency(sim_df["total_cost"].mean()))
        st.plotly_chart(px.histogram(sim_df, x="avg_service_level", nbins=25), use_container_width=True)
        st.plotly_chart(px.box(sim_df, y="total_cost"), use_container_width=True)

