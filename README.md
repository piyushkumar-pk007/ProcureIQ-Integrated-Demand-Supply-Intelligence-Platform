# ProcureIQ — Integrated Demand-Supply Intelligence Platform

ProcureIQ is an end-to-end supply chain analytics project built for a data science / supply chain analytics interview setting. The goal is to show how demand forecasting, supplier risk scoring, procurement optimization, and scenario simulation can work together in one decision-support workflow instead of living in separate notebooks.

**This is a Kaggle-based interview demonstration inspired by real supply-chain procurement and forecasting problems.**

The project uses the public Kaggle dataset `apoorvwatsky/supply-chain-shipment-pricing-data`. It is not production client data, and it should be presented honestly as a prototype that demonstrates consulting-style problem solving, modeling choices, and implementation quality.

## What this project is

At a high level, ProcureIQ answers three practical business questions:

1. What are we likely to need next?
2. Which suppliers are operationally risky?
3. Given cost, risk, and service constraints, how should we allocate procurement?

To answer those questions, the project is organized into six connected layers:

1. Data ingestion
2. Data cleaning and preprocessing
3. Feature engineering
4. Forecasting
5. Supplier risk scoring
6. Procurement optimization and scenario simulation

The final output is a Streamlit dashboard that makes the analysis easier to explain in an interview or portfolio review.

## Why this matters

In supply chain and procurement analytics, forecasting alone is not enough. A forecast can say demand is rising, but that does not automatically tell you:

- whether your current suppliers are reliable
- whether freight behavior is becoming unstable
- whether concentration risk is growing
- whether the cheapest supplier is actually the best choice once service-level risk is included

This is why the project is built as a decision intelligence workflow rather than just a forecasting notebook.

From a consulting point of view, this matters because stakeholders usually care about business decisions, not isolated model accuracy. A stronger interview story is:

- demand was forecasted
- supplier behavior was scored
- sourcing decisions were optimized
- recommendations were stress-tested under uncertainty

That is much closer to how a real analytics solution would be framed in Strategy & Consulting, Data & AI, or supply chain transformation work.

## What data was used

**Dataset source**

- Kaggle: `apoorvwatsky/supply-chain-shipment-pricing-data`

**Expected business fields**

The dataset contains procurement and shipment-related fields such as:

- country
- vendor
- manufacturing site
- product group
- sub classification
- item description
- brand
- vendor inco term
- fulfill via
- shipment mode
- PO sent to vendor date
- scheduled delivery date
- delivered to client date
- delivery recorded date
- unit of measure
- line item quantity
- line item value
- pack price
- unit price
- weight
- freight cost
- insurance cost

## What the project does

### 1. Data ingestion

The ingestion layer supports two paths:

- automatic Kaggle API download if credentials are available in `.env`
- manual CSV placement inside `data/raw/`

It detects the main CSV, reads it safely, and saves an intermediate copy to:

- `data/processed/clean_base.csv`

This makes the rest of the pipeline independent of how the data was originally obtained.

### 2. Preprocessing

Operational data is messy. This project does not assume a clean source table.

The preprocessing step:

- standardizes column names to `snake_case`
- parses shipment and delivery dates
- converts key numeric fields
- handles text placeholders such as:
  - `Freight Included in Commodity Cost`
  - `Invoiced Separately`
  - `Weight Captured Separately`
  - `See DN...`
  - `Date Not Captured`
  - `N/A - From RDC`
- creates missing-value flags before imputation
- logs cleaning decisions
- avoids silently dropping rows

In the current run of this project:

- rows before cleaning: `10,324`
- rows after cleaning: `10,324`
- row drop rate: `0.0%`

That is important because it shows the pipeline preserved the original observations while still improving usability for modeling.

### 3. Feature engineering

After preprocessing, the project creates features that are useful for both business analysis and modeling.

Examples include:

- `delivery_delay_days`
- `is_late_delivery`
- `verification_delay_days`
- `total_landed_cost`
- `freight_per_kg`
- `value_per_unit`
- `weighted_unit_price`
- `lead_time_days`
- `vendor_volume_share`
- `vendor_price_cv`
- `vendor_avg_delay`
- `vendor_late_rate`
- `item_weekly_demand`
- `item_weekly_avg_price`

These features matter because they convert raw transactions into signals that can drive better forecasting, risk scoring, and optimization decisions.

### 4. Forecasting

The forecasting layer predicts:

- weekly demand using `sum(line_item_quantity)`
- weekly weighted average unit price

It does not rely on a single model. Instead, it compares several methods:

- seasonal naive baseline
- moving average baseline
- SARIMA
- Prophet if available
- XGBoost, with RandomForest fallback if needed

**Why these models were used**

- `seasonal_naive` is a simple baseline that every time-series project should beat.
- `moving_average` is useful when series are noisy and the user wants a transparent benchmark.
- `SARIMA` is a strong classical model for structured univariate time series.
- `Prophet` is optional because it is easy to explain, but not always necessary.
- `XGBoost` is included because lag-based tree models often work well when demand patterns are nonlinear or irregular.

**Why walk-forward validation was used**

Random train/test splitting is not appropriate for time series because it leaks future information. This project uses time-aware validation so model quality is measured in a way that resembles real planning cycles.

**What happened in the current run**

The strongest results in this run came from the weighted-average price forecasts for HRDT item series.

Examples from `data/outputs/model_comparison.csv`:

- `HRDT | HRDT | HIV 1/2, Uni-Gold HIV Kit, 20 Tests`
  - target: `weekly_weighted_avg_price`
  - best visible model: `xgboost_or_rf`
  - WAPE: `3.57`
  - RMSE: `0.235`
- `HRDT | HRDT | HIV 1/2, Determine Complete HIV Kit, 100 Tests`
  - target: `weekly_weighted_avg_price`
  - best visible model: `xgboost_or_rf`
  - WAPE: `7.88`
  - RMSE: `0.115`

The project is also designed to fall back to a coarser aggregation if item-level weekly history is too sparse. That keeps the work honest and practical for public data.

Forecast outputs are saved to:

- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_results.csv`
- forecast plot PNGs in `data/outputs/`

### 5. Supplier risk scoring

The supplier risk module creates an interpretable risk score from `0` to `100`, where:

- `0` means very safe
- `100` means very risky

It uses two approaches:

**A. Rule-based business score**

Built from:

- on-time delivery rate
- average delay
- delay volatility
- price volatility
- freight volatility
- missing-data rate
- shipment concentration
- country and product dependence where possible

**B. ML late-delivery risk model**

A tree-based classifier predicts whether a shipment will be late using historical operational features. Feature importance is saved, and SHAP can be added if available.

**Why this design was used**

The rule-based score is easy to explain to procurement stakeholders. The ML model adds predictive signal and shows stronger data science depth. Together, they balance explainability and modeling sophistication.

**What happened in the current run**

- suppliers scored: `73`
- most suppliers landed in the `Medium Risk` band

Examples from `data/outputs/top_risky_suppliers.csv`:

- `SCMS from RDC` with risk score `64.19`
- `BIO-RAD LABORATORIES (FRANCE)` with risk score `51.19`
- `Aurobindo Pharma Limited` with risk score `50.24`
- `CIPLA LIMITED` with risk score `49.91`

Outputs are saved to:

- `data/outputs/supplier_risk_scores.csv`
- `data/outputs/top_risky_suppliers.csv`
- `data/outputs/supplier_risk_distribution.png`
- `data/outputs/supplier_risk_feature_importance.png`

### 6. Procurement optimization

The optimization layer translates analytics into a recommendation.

Decision variable:

- how much of each item to procure from each historically feasible vendor

Objective:

- minimize purchase cost
- minimize freight cost
- include holding cost
- penalize supplier risk
- satisfy demand and service constraints

The model is built with **PuLP**.

**Why PuLP was used**

- it is easy to explain in interviews
- it is Python-native
- it supports clear business constraints
- it is appropriate for a portfolio-grade mixed-integer optimization demonstration

**Constraints included**

- demand satisfaction
- minimum order quantity
- storage capacity
- risky supplier allocation caps
- ABC service-level targets

**Why historical vendor-item pairs were used**

The Kaggle data does not explicitly define a perfect supplier-item master. So feasible vendor-item combinations are derived from historical shipments. That is a realistic and transparent way to avoid recommending impossible allocations.

**What happened in the current run**

From `data/outputs/optimization_summary.csv`:

- optimized total cost: `242.39M`
- cheapest-supplier-only baseline: `578.50M`
- historical-allocation baseline: `612.99M`

From `data/outputs/cost_savings_vs_baseline.csv`:

- savings vs cheapest-only baseline: `336.10M`
- savings vs historical-allocation baseline: `370.59M`

These savings are generated from a prototype model with public data and configurable assumptions, so they should be described as scenario outputs rather than guaranteed real-world savings.

### 7. Scenario simulation

The final analytical layer tests how the optimized plan behaves under uncertainty.

Scenarios include:

- base case
- high demand case
- supplier delay case
- cost inflation case

This is done with Monte Carlo simulation.

**Why simulation was used**

Optimization produces a point recommendation, but procurement decisions live in uncertainty. Simulation helps answer:

- what happens if demand spikes?
- what happens if risky suppliers are delayed?
- what happens if costs inflate?

**What happened in the current run**

From `data/outputs/scenario_summary.csv`:

- base case average service level: `0.907`
- supplier delay case average service level: `0.909`
- high demand case average service level: `0.782`
- cost inflation case average total cost: `261.79M`

The most important business takeaway from this run is that the plan is much more vulnerable to demand shock than to the modeled supplier delay shock.

## Why these tools and libraries were used

### Python

Python is the main implementation language because it supports the full workflow:

- data preparation
- modeling
- optimization
- testing
- dashboarding

### Pandas and NumPy

Used for:

- tabular cleaning
- aggregations
- date handling
- feature generation

They are standard choices for structured supply chain analytics work.

### Statsmodels

Used for SARIMA because it is one of the most direct and explainable libraries for classical time-series forecasting.

### XGBoost and scikit-learn

Used for:

- lag-feature forecasting
- late-delivery classification
- standard validation and metrics

These tools are practical, widely recognized, and strong enough for an interview-quality demonstration.

### PuLP

Used for optimization because it is readable and business-friendly.

### Streamlit

Used for the dashboard because it is fast to build, easy to run locally, and well suited for analytics demos.

### Pytest

Used so the repo feels like a real engineering project instead of a notebook-only project.

## Repository structure

```text
procureiq-demand-supply-intelligence/
├── run_project.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── notebooks/
│   ├── 01_eda_data_quality.ipynb
│   ├── 02_forecasting_experiments.ipynb
│   ├── 03_supplier_risk_scoring.ipynb
│   └── 04_procurement_optimization.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── supplier_risk.py
│   ├── optimization.py
│   ├── simulation.py
│   ├── dashboard_data.py
│   ├── config.py
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── reports/
├── tests/
└── assets/
```

## Architecture

```text
Kaggle CSV / Manual CSV
        |
        v
data_ingestion.py
        |
        v
preprocessing.py
        |
        v
feature_engineering.py
        |
        +--------------------+------------------------+----------------------+
        |                    |                        |                      |
        v                    v                        v                      v
forecasting.py         supplier_risk.py        optimization.py        simulation.py
        |                    |                        |                      |
        +--------------------+------------------------+----------------------+
                                     |
                                     v
                             Streamlit dashboard
```

An image version is available at `assets/architecture_diagram.png`.

## How to use this project

### Step 1. Open the project

```powershell
cd "f:\Portfolio\Supply Chain - DS Project\procureiq-demand-supply-intelligence"
```

### Step 2. Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks it, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
```

### Step 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5. Set up Kaggle credentials

```powershell
copy .env.example .env
```

Then fill in:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_DATASET=apoorvwatsky/supply-chain-shipment-pricing-data`

If you do not want to use Kaggle API:

- manually download the dataset
- place the CSV inside `data/raw/`

### Step 6. Run the entire pipeline

The easiest option is:

```powershell
python run_project.py
```

This runs the following steps in order:

1. data ingestion
2. preprocessing
3. feature engineering
4. forecasting
5. supplier risk
6. optimization
7. simulation

### Step 7. Open the dashboard

```powershell
python -m streamlit run app\streamlit_app.py
```

### Step 8. Run tests

```powershell
python -m pytest
```

## Manual run option

If you prefer to execute the pipeline one step at a time:

```powershell
python -m src.data_ingestion
python -m src.preprocessing
python -m src.feature_engineering
python -m src.forecasting
python -m src.supplier_risk
python -m src.optimization
python -m src.simulation
```

Important:

- use `python -m ...`
- do not run them as `python src\file.py`

## How to use `run_project.py`

### Run everything

```powershell
python run_project.py
```

### Resume from a later step

Example: start from forecasting

```powershell
python run_project.py --from-step 4
```

### Stop after a specific step

Example: run only through supplier risk

```powershell
python run_project.py --to-step 5
```

### Run tests after the pipeline

```powershell
python run_project.py --include-tests
```

### Run the pipeline and then open the dashboard

```powershell
python run_project.py --open-dashboard
```

### List available steps

```powershell
python run_project.py --list-steps
```

## Files created by the pipeline

### Processed data

- `data/processed/clean_base.csv`
- `data/processed/raw_source_reference.csv`
- `data/processed/cleaned_supply_chain.csv`
- `data/processed/cleaning_summary.csv`
- `data/processed/engineered_supply_chain.csv`
- `data/processed/weekly_snapshot.csv`

### Forecasting outputs

- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_results.csv`
- forecast PNG files in `data/outputs/`

### Supplier risk outputs

- `data/outputs/supplier_risk_scores.csv`
- `data/outputs/top_risky_suppliers.csv`
- `data/outputs/supplier_risk_distribution.png`
- `data/outputs/supplier_risk_feature_importance.png`

### Optimization outputs

- `data/outputs/recommended_order_plan.csv`
- `data/outputs/optimization_summary.csv`
- `data/outputs/cost_savings_vs_baseline.csv`
- `data/outputs/supplier_allocation_chart.png`

### Simulation outputs

- `data/outputs/service_level_distribution.csv`
- `data/outputs/expected_stockout_cost.csv`
- `data/outputs/scenario_summary.csv`

## Dashboard pages

The Streamlit app includes five pages:

### Executive Overview

Used to summarize:

- total shipments
- total landed cost
- late delivery rate
- forecast accuracy
- potential savings

### Forecasting

Used to:

- select a product group or item
- compare actuals vs forecasts
- review model comparisons

### Supplier Risk

Used to:

- inspect supplier scores
- identify risky vendors
- understand the main drivers of risk

### Procurement Optimizer

Used to:

- adjust service level, risk weight, and storage assumptions
- review recommended allocations
- compare cost outcomes vs baselines

### Scenario Simulator

Used to:

- stress test the recommended plan
- compare demand shock, delay shock, and inflation effects

## Interview talking points

### 30-second version

I built ProcureIQ, a Kaggle-based supply chain intelligence prototype that combines demand and price forecasting, supplier risk scoring, and procurement optimization. It forecasts weekly demand and unit price, identifies risky suppliers using delivery and price volatility signals, and recommends optimized supplier allocation using PuLP. The system demonstrates how data science can support S&OP, procurement, and inventory decisions.

### 2-minute version

This project shows how I would connect data engineering, predictive modeling, optimization, and decision support in a supply chain consulting context. I start with public shipment pricing and delivery data, clean and standardize operational fields, engineer vendor and item-level features, and forecast demand and price using time-aware validation. I then score supplier risk from reliability, volatility, and concentration patterns, and use those signals inside a procurement optimizer built with PuLP. Finally, I stress-test the recommended sourcing plan under demand and cost uncertainty and expose the outputs through a Streamlit dashboard. The point of the project is not just to build a model, but to build a business decision workflow.

## Common interviewer questions

### Why use XGBoost if ARIMA is available?

Because they solve slightly different problems well. ARIMA is a very good structured baseline for univariate time series. XGBoost is useful when lagged features, rolling windows, and nonlinear behavior matter. Comparing both is stronger than assuming one should always win.

### Why walk-forward validation?

Because time-series performance should be measured in time order. Random K-fold would leak future information and make the evaluation less realistic.

### How was supplier risk calculated?

The score combines late delivery behavior, average delay, delay volatility, price volatility, freight volatility, missing-data rate, and concentration effects. That produces an explainable risk number from 0 to 100. A separate ML model predicts late shipment probability.

### Why PuLP?

Because it is readable, practical, and easy to explain. For an interview project, it shows optimization thinking without hiding the business logic behind a black box.

### What are the optimization constraints?

Demand satisfaction, MOQ, storage capacity, risky supplier caps, and service-level targets by item class.

### How would this be deployed on Oracle OCI Data Science?

Raw and processed data would move to OCI Object Storage. Model development could happen in OCI Data Science notebooks or jobs. Trained models could be tracked in OCI Model Catalog. Scheduled scoring and optimization could run through OCI Functions or OCI Data Flow. Results could be served via Autonomous Database and exposed in a separate dashboard layer.

### How would this scale with PySpark?

Preprocessing, feature engineering, and group-level aggregations are the first candidates to move to PySpark DataFrames. Forecasting could remain selective at the grouped-series level, while the feature tables and shipment processing scale out in Spark.

## Honest project framing

Use these points clearly in the interview:

- this is a public-data prototype, not production enterprise data
- some planning parameters such as MOQ, service level, and storage capacity are configurable assumptions
- the cost savings are model outputs from a scenario-based optimization exercise, not audited real savings
- the purpose of the project is to demonstrate end-to-end analytics thinking, clean implementation, and business communication

## What to screenshot for your portfolio or interview

- one EDA view showing missing values or vendor concentration
- one forecast chart with actual vs forecast
- one model comparison table
- one supplier risk table or distribution
- one optimization cost comparison view
- one scenario summary or service-level distribution
- one dashboard screenshot from each major page

## Final note

ProcureIQ is strongest when it is presented as a decision-support prototype, not just a coding exercise. The value of the project is that it shows how data engineering, forecasting, supplier analytics, optimization, and simulation can be combined into one coherent business workflow.
