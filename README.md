# ProcureIQ — Integrated Demand-Supply Intelligence Platform

ProcureIQ is a supply chain analytics project that brings together forecasting, supplier performance analytics, procurement optimization, and scenario simulation in one workflow. The idea behind the project is simple: supply chain decisions are rarely about only one thing. A planning team may know demand is rising, but they still need to understand which suppliers are reliable, how freight and pricing are behaving, and how to allocate orders under practical business constraints.

This project was built to answer those questions in a connected way rather than as isolated analyses.

**This is a Kaggle-based demonstration inspired by real supply-chain procurement and forecasting problems.**

The dataset used here is public Kaggle data, not production enterprise data. That matters, because some business constraints such as MOQ policy, storage limits, and service-level assumptions are modeled explicitly in the code rather than coming directly from a live ERP or procurement system.

## What this project is

ProcureIQ is an end-to-end decision-support project for procurement and supply chain planning. It starts with shipment-level procurement data and moves through the full analytical lifecycle:

1. ingest and clean operational data
2. build supply chain features
3. forecast demand and unit price
4. score supplier reliability and risk
5. optimize procurement allocation
6. simulate stress scenarios
7. present results through a dashboard

The project is designed to answer a practical planning question:

**If demand changes, supplier behavior is uneven, and cost matters, how should procurement decisions be made more intelligently?**

## Why this matters

In real supply chain environments, a forecast is only part of the story.

If the team forecasts demand well but ignores supplier risk:

- stockouts can still happen
- late deliveries can still disrupt service levels
- freight cost instability can still hurt margins
- procurement may over-concentrate on a single supplier

If the team optimizes only for cheapest cost:

- service risk may increase
- late suppliers may receive too much allocation
- demand uncertainty may not be considered

This project matters because it treats supply chain planning as a connected decision problem rather than a single-model exercise.

## Business problem

The business problem behind ProcureIQ is common across procurement and planning functions:

- demand is uncertain
- suppliers are not equally reliable
- pricing and freight behavior vary over time
- planners need service-level protection
- procurement teams need an allocation plan, not just a report

The goal of the platform is to turn raw historical shipment data into decisions that are more actionable:

- what demand is likely next
- which suppliers look risky
- how sourcing should be allocated
- what could happen under adverse scenarios

## Dataset

Source:

- Kaggle: `apoorvwatsky/supply-chain-shipment-pricing-data`

The data includes fields such as:

- country
- vendor
- manufacturing site
- product group
- item description
- shipment mode
- PO sent date
- scheduled delivery date
- delivered date
- recorded date
- line item quantity
- line item value
- pack price
- unit price
- weight
- freight cost
- insurance cost

This is enough to build a meaningful prototype around planning, supplier performance, and sourcing logic.

## Project approach

The project is organized into a few major layers.

### 1. Data ingestion

The first step is to make data loading easy and reliable.

ProcureIQ supports:

- Kaggle API download through `.env`
- manual CSV placement in `data/raw/`

The ingestion step detects the source file automatically and saves an intermediate version to:

- `data/processed/clean_base.csv`

This avoids repeating the raw download or manual placement logic later in the pipeline.

### 2. Preprocessing

Supply chain datasets often contain inconsistent date formats, placeholder text, mixed numeric fields, and incomplete records. This dataset had the same kind of issues.

The preprocessing step:

- standardizes columns to `snake_case`
- parses operational date fields
- converts core numeric columns
- creates missing-value flags before imputation
- handles text placeholders such as:
  - `Freight Included in Commodity Cost`
  - `Invoiced Separately`
  - `Weight Captured Separately`
  - `See DN...`
  - `Date Not Captured`
  - `N/A - From RDC`
- logs cleaning decisions

One important choice here was to avoid quietly dropping records. The pipeline keeps the row count intact and makes the cleaning traceable.

Current run summary:

- rows before cleaning: `10,324`
- rows after cleaning: `10,324`
- row drop rate: `0.0%`

That was intentional. In operational analytics, silent record loss can distort vendor performance and demand patterns.

### 3. Feature engineering

After preprocessing, the project creates features that make the data useful for both analysis and modeling.

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

These features are important because they translate raw transactions into procurement signals:

- delay behavior
- landed cost behavior
- vendor consistency
- demand evolution over time

### 4. Forecasting

The forecasting layer predicts two things:

- weekly demand using `sum(line_item_quantity)`
- weekly weighted average unit price

Instead of relying on one model, the project compares multiple approaches:

- seasonal naive
- moving average
- SARIMA
- Prophet if installed
- XGBoost, with RandomForest fallback

### Why these models were chosen

Each model was chosen for a reason:

- `seasonal_naive` is a baseline that any serious forecast should be compared against
- `moving_average` is simple, transparent, and useful when explaining results to non-technical stakeholders
- `SARIMA` provides a strong classical time-series benchmark
- `Prophet` is optional because it is easy to use and compare, but not required
- `XGBoost` is included because lag-based gradient boosting often performs well when behavior is nonlinear or noisy

The point is not to force one “best” technique. The point is to compare approaches honestly and keep the winning model evidence-based.

### Why walk-forward validation was used

Time series should be validated in time order. Random train/test splitting would leak future information and make results look better than they really are. For that reason, the project uses walk-forward style validation through `TimeSeriesSplit`.

### Current forecasting outcome

In the current run, some of the strongest results came from weighted-average price forecasts for HRDT item series.

Examples from `data/outputs/model_comparison.csv`:

- `HRDT | HRDT | HIV 1/2, Uni-Gold HIV Kit, 20 Tests`
  - target: `weekly_weighted_avg_price`
  - model: `xgboost_or_rf`
  - WAPE: `3.57`
  - RMSE: `0.235`
- `HRDT | HRDT | HIV 1/2, Determine Complete HIV Kit, 100 Tests`
  - target: `weekly_weighted_avg_price`
  - model: `xgboost_or_rf`
  - WAPE: `7.88`
  - RMSE: `0.115`

The code also handles a practical public-data limitation: if item-level weekly history is not deep enough, the pipeline can aggregate more coarsely instead of pretending the series is stronger than it is.

Forecast outputs are saved to:

- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_results.csv`
- forecast plot PNGs in `data/outputs/`

### 5. Supplier risk scoring

Supplier risk is modeled in two layers.

#### Rule-based supplier risk score

This produces a score from `0` to `100`, where:

- `0` means very safe
- `100` means very risky

It is based on:

- on-time delivery rate
- average delay
- delay volatility
- price volatility
- freight volatility
- missing-data rate
- country concentration
- product concentration

#### ML-based late delivery model

In addition to the rule-based score, the project trains a tree-based classifier to estimate late-delivery likelihood from historical shipment features.

### Why both approaches were used

The rule-based score is useful because business users can understand it quickly. The ML model is useful because it can detect nonlinear patterns and gives a predictive second view of supplier reliability.

This combination was chosen deliberately:

- rule-based scoring supports explainability
- ML supports predictive depth

### Current supplier risk outcome

From the current run:

- suppliers scored: `73`

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

The optimization layer turns the analysis into a sourcing recommendation.

The decision problem is:

**How much should be procured from each supplier for each item while balancing cost, risk, and service requirements?**

The optimization model is built with **PuLP**.

The objective combines:

- purchase cost
- freight cost
- holding cost
- supplier risk penalty

The constraints include:

- demand satisfaction
- minimum order quantity
- storage capacity
- risky supplier allocation caps
- ABC-based service-level logic

### Why PuLP was chosen

PuLP was chosen because it is readable, transparent, and practical for this kind of prototype. It makes the decision logic easy to inspect, which is important when the audience includes both technical and business stakeholders.

### Why historical vendor-item pairs were used

The dataset does not provide a complete vendor-item master with formal sourcing eligibility rules. To keep recommendations realistic, the optimizer only allows vendor-item combinations that actually appeared in historical shipments.

That is a simple but important design choice. Without it, the optimizer could recommend theoretically cheap but operationally unsupported supplier-item allocations.

### Current optimization outcome

From `data/outputs/optimization_summary.csv`:

- optimized total cost: `242.39M`
- cheapest-supplier-only baseline: `578.50M`
- historical-allocation baseline: `612.99M`

From `data/outputs/cost_savings_vs_baseline.csv`:

- estimated improvement vs cheapest-only baseline: `336.10M`
- estimated improvement vs historical allocation baseline: `370.59M`

These should be presented as model-based scenario outputs, not guaranteed realized savings.

### 7. Scenario simulation

The simulation layer stress-tests the optimized plan.

Scenarios included:

- base case
- high demand case
- supplier delay case
- cost inflation case

Monte Carlo simulation is used because a procurement plan should be tested under uncertainty, not just evaluated at one point estimate.

### Current simulation outcome

From `data/outputs/scenario_summary.csv`:

- base case average service level: `0.907`
- supplier delay case average service level: `0.909`
- high demand case average service level: `0.782`
- cost inflation case average total cost: `261.79M`

The most useful takeaway from this run is that demand shock was more damaging to service level than the modeled supplier delay shock. That suggests the next design improvement should likely focus on inventory buffers, demand sensing, or service-level policy rather than only supplier diversification.

## Problems faced and how they were handled

This project was not built on perfectly analysis-ready data. A few practical problems came up, and they shaped the design.

### Problem 1: messy operational fields

The dataset contains placeholder text in places where numeric or date values would normally be expected.

Examples:

- freight embedded in commodity cost
- weight captured separately
- missing dates
- note-style text fields such as `See DN...`

**How it was handled**

- standardized placeholder handling in preprocessing
- created missing flags before imputation
- logged cleaning decisions rather than hiding them

### Problem 2: mixed date and encoding issues

The CSV required more careful loading than a clean UTF-8 file, and date parsing was not uniform across fields.

**How it was handled**

- added encoding fallback during ingestion
- converted date parsing into a controlled preprocessing step
- kept source and processed layers separate

### Problem 3: not every series has enough time depth

Public datasets often look rich at the transaction level but become thin after item-level time aggregation.

**How it was handled**

- filtered for higher-history series
- used baseline and classical models alongside ML
- allowed fallback aggregation when needed instead of pretending the data was deeper than it was

### Problem 4: procurement constraints are not fully explicit in public data

Real procurement systems often contain business rules that do not appear directly in Kaggle datasets.

**How it was handled**

- modeled configurable assumptions in `src/config.py`
- constrained optimization to historically observed vendor-item combinations
- made assumptions visible in code and documentation

### Problem 5: analytics is not enough without decisions

Forecasts and risk tables are useful, but teams still need an actual order recommendation.

**How it was handled**

- connected forecasting and supplier risk into a procurement optimization layer
- added simulation to test whether the recommended plan holds up under shocks

## How this can be used

This project can be used in several ways.

### As a planning prototype

A supply chain or procurement team could use it to:

- monitor item demand trends
- compare price behavior over time
- review supplier reliability
- generate sourcing recommendations

### As a dashboard for business discussion

The Streamlit app can support conversations such as:

- which items are becoming more expensive
- which suppliers should be watched more closely
- how much cost and risk change when allocation rules change

### As a technical project foundation

The repository structure is modular enough to extend into:

- cloud deployment
- scheduled scoring
- larger data processing frameworks such as PySpark
- database-backed dashboards

## Why the final outcome matters

The value of ProcureIQ is not only that it produces forecasts or risk scores. The real value is that it connects those outputs into a sourcing recommendation that can be stress-tested.

In the current run, the project produced:

- cleaned and engineered supply chain data
- forecast comparison outputs
- supplier risk scores for `73` suppliers
- an optimized procurement plan
- scenario-based service-level and cost outputs
- a working Streamlit dashboard

That makes the final outcome more complete than a notebook-based analysis. It behaves more like a small decision-support product.

## Project structure

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
├── src/
├── app/
├── reports/
├── tests/
└── assets/
```

## How to run the project

### Step 1: move into the project folder

```powershell
cd "f:\Portfolio\Supply Chain - DS Project\procureiq-demand-supply-intelligence"
```

### Step 2: create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
```

### Step 4: install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5: set up the dataset

Option A: Kaggle API

```powershell
copy .env.example .env
```

Then fill in:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_DATASET=apoorvwatsky/supply-chain-shipment-pricing-data`

Option B: manual dataset placement

- download the CSV manually
- place it inside `data/raw/`

### Step 6: run the full pipeline

The easiest way:

```powershell
python run_project.py
```

### Step 7: open the dashboard

```powershell
python -m streamlit run app\streamlit_app.py
```

### Step 8: run tests

```powershell
python -m pytest
```

## Run steps one by one

If you want to run everything manually:

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
- do not run the files as `python src\file.py`

## Using the runner script

### Run all pipeline steps

```powershell
python run_project.py
```

### Start from a later step

Example: start from forecasting

```powershell
python run_project.py --from-step 4
```

### Stop after a specific step

Example: stop after supplier risk scoring

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

### List all steps

```powershell
python run_project.py --list-steps
```

## Output files

### Processed files

- `data/processed/clean_base.csv`
- `data/processed/raw_source_reference.csv`
- `data/processed/cleaned_supply_chain.csv`
- `data/processed/cleaning_summary.csv`
- `data/processed/engineered_supply_chain.csv`
- `data/processed/weekly_snapshot.csv`

### Forecasting files

- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_results.csv`
- forecast PNG charts

### Supplier risk files

- `data/outputs/supplier_risk_scores.csv`
- `data/outputs/top_risky_suppliers.csv`
- risk distribution and feature importance plots

### Optimization files

- `data/outputs/recommended_order_plan.csv`
- `data/outputs/optimization_summary.csv`
- `data/outputs/cost_savings_vs_baseline.csv`
- `data/outputs/supplier_allocation_chart.png`

### Simulation files

- `data/outputs/service_level_distribution.csv`
- `data/outputs/expected_stockout_cost.csv`
- `data/outputs/scenario_summary.csv`

## Dashboard pages

The Streamlit app includes:

- Executive Overview
- Forecasting
- Supplier Risk
- Procurement Optimizer
- Scenario Simulator

## Technology choices

### Python

Used as the core language because it supports data processing, forecasting, optimization, testing, and dashboarding in one ecosystem.

### Pandas and NumPy

Used for data cleaning, aggregations, and feature creation.

### Statsmodels

Used for SARIMA forecasting because it is a strong classical benchmark.

### XGBoost and scikit-learn

Used for lag-based forecasting and late-delivery prediction because they handle nonlinear patterns well and are widely trusted in applied ML workflows.

### PuLP

Used for optimization because it is transparent and easy to translate into business logic.

### Streamlit

Used to make the outputs easier to explore and present.

### Pytest

Used to keep the project structured and testable.

## Final takeaway

ProcureIQ shows how forecasting, supplier analytics, optimization, and simulation can be combined into one supply chain decision workflow. The project is not just about predicting the future. It is about using historical operational data to make better procurement choices, understand trade-offs, and make planning decisions more resilient.
