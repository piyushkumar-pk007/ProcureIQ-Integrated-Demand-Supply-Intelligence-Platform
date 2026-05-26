# ProcureIQ — Integrated Demand-Supply Intelligence Platform

ProcureIQ is an interview-ready supply chain analytics project that combines demand forecasting, supplier risk scoring, procurement optimization, and scenario simulation into one modular Python repository.

**This is a Kaggle-based interview demonstration inspired by real supply-chain procurement and forecasting problems.**

## Why this matters in supply chain consulting
- Procurement teams need a joined-up view of demand, price, supplier reliability, and sourcing constraints.
- Forecasting alone is not enough if risky suppliers create lead-time shocks or freight volatility.
- Optimization turns analytics into a decision recommendation, which is closer to how consulting teams create measurable business value.

## Dataset
- Source: Kaggle, `apoorvwatsky/supply-chain-shipment-pricing-data`
- Expected file location if downloaded manually: `data/raw/`
- The repository does not ship with the dataset to stay compliant with public-data licensing.

## Business objective
Build a decision intelligence workflow with three connected layers:

1. Forecast weekly demand and weighted average unit price for high-signal product/item series.
2. Score supplier risk using delivery, price, freight, and consistency signals.
3. Optimize supplier allocation and order quantity under business constraints.

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
├── app/
├── reports/
├── tests/
└── assets/
```

## Architecture
Text diagram:

```text
Kaggle CSV / Manual CSV
        |
        v
data_ingestion.py
        |
        v
preprocessing.py --> cleaned_supply_chain.csv
        |
        v
feature_engineering.py --> engineered_supply_chain.csv
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

If `assets/architecture_diagram.png` is present, use that in GitHub as the visual architecture asset.

## Modules

### 1. Data preprocessing
- Detects the source CSV automatically.
- Supports Kaggle API download when credentials exist.
- Supports a manual CSV drop into `data/raw/`.
- Standardizes columns to `snake_case`.
- Parses operational date fields.
- Creates missing-value flags before imputation.
- Converts messy numeric fields and logs cleaning decisions.

### 2. Forecasting
- Forecast targets:
  - weekly demand: `sum(line_item_quantity)`
  - weekly weighted average unit price
- Validation uses walk-forward / `TimeSeriesSplit`, not random K-fold.
- Models:
  - seasonal naive
  - moving average
  - SARIMA
  - Prophet, if installed
  - XGBoost regressor, with RandomForest fallback if XGBoost is unavailable
- Output files:
  - `data/outputs/model_comparison.csv`
  - `data/outputs/forecast_results.csv`
  - forecast PNGs in `data/outputs/`

If the dataset lacks enough clean weekly history for item-level forecasts, the code automatically falls back to `product_group + month` aggregation. That tradeoff is intentional and is logged so the project stays honest about public-data limitations.

Because the public Kaggle data is not checked into this repository, the exact best model is generated after you run the pipeline locally. In most interview discussions, I position the model choice as evidence-driven: whichever model wins on walk-forward WAPE/RMSE in `model_comparison.csv` becomes the recommended production candidate.

### 3. Supplier risk
- Rule-based score from 0 to 100, where 0 is safest and 100 is riskiest.
- Risk inputs:
  - on-time delivery rate
  - average delay
  - delay volatility
  - price volatility
  - shipment volume
  - freight volatility
  - missing-data rate
  - concentration risk
- ML risk layer predicts late delivery using tree-based classification.
- SHAP is optional; feature-importance fallback is included.

### 4. Procurement optimization
- Uses PuLP mixed-integer optimization.
- Objective minimizes:
  - purchase cost
  - freight cost
  - holding cost
  - supplier risk penalty
  - implicit stockout exposure via demand/service-level constraints
- Constraints include:
  - demand satisfaction
  - storage capacity
  - minimum order quantity
  - risky supplier allocation cap
  - ABC-based service levels

### 5. Simulation
- Monte Carlo simulation with 500+ iterations.
- Scenarios:
  - base case
  - high demand case
  - supplier delay case
  - cost inflation case

### 6. Dashboard
- Executive Overview
- Forecasting Explorer
- Supplier Risk page
- Procurement Optimizer page
- Scenario Simulator page

## Key metrics generated
- MAPE, WAPE, RMSE, MAE for forecasting
- supplier risk score and risk band
- late-delivery probability from ML classifier
- optimized procurement cost
- savings versus cheapest-supplier and historical-allocation baselines
- simulated service-level distribution and expected stockout cost

## How to run

### Step 1. Open the project folder
```powershell
cd "f:\Portfolio\Supply Chain - DS Project\procureiq-demand-supply-intelligence"
```

### Step 2. Create a virtual environment
```powershell
python -m venv .venv
```

### Step 3. Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:
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
Then open `.env` and add:
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `KAGGLE_DATASET=apoorvwatsky/supply-chain-shipment-pricing-data`

If you do not want to use the Kaggle API:
- manually download the dataset CSV
- place it inside `data/raw/`

### Step 6. Run everything using one command
This is the easiest option for beginners:

```powershell
python run_project.py
```

That script runs these steps in order:
1. data ingestion
2. preprocessing
3. feature engineering
4. forecasting
5. supplier risk scoring
6. procurement optimization
7. scenario simulation

### Step 7. Launch the dashboard
```powershell
python -m streamlit run app\streamlit_app.py
```

### Step 8. Run tests
```powershell
python -m pytest
```

## Alternative: run each pipeline step manually
Use these commands if you want to run one step at a time:

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
- run the modules with `python -m ...`
- do not run them as `python src\file.py`

## How to use `run_project.py`

### Run the full pipeline
```powershell
python run_project.py
```

### Resume from a later step
Example: start from forecasting
```powershell
python run_project.py --from-step 4
```

### Stop after a certain step
Example: run only through supplier risk
```powershell
python run_project.py --to-step 5
```

### Run the pipeline and tests together
```powershell
python run_project.py --include-tests
```

### Run the pipeline and open the dashboard after it finishes
```powershell
python run_project.py --open-dashboard
```

### Show available steps
```powershell
python run_project.py --list-steps
```

## Output files created
- `data/processed/clean_base.csv`
- `data/processed/cleaned_supply_chain.csv`
- `data/processed/engineered_supply_chain.csv`
- `data/outputs/model_comparison.csv`
- `data/outputs/forecast_results.csv`
- `data/outputs/supplier_risk_scores.csv`
- `data/outputs/top_risky_suppliers.csv`
- `data/outputs/recommended_order_plan.csv`
- `data/outputs/optimization_summary.csv`
- `data/outputs/cost_savings_vs_baseline.csv`
- `data/outputs/service_level_distribution.csv`
- `data/outputs/expected_stockout_cost.csv`
- `data/outputs/scenario_summary.csv`

## Interview pitch

### 30-second version
I built ProcureIQ, a Kaggle-based supply chain intelligence prototype that combines demand and price forecasting, supplier risk scoring, and procurement optimization. It forecasts weekly demand and unit price, identifies risky suppliers using delivery and price volatility signals, and recommends optimized supplier allocation using PuLP. The system demonstrates how data science can support S&OP, procurement, and inventory decisions.

### 2-minute version
This project shows how I would connect data engineering, predictive modeling, optimization, and decision support in a supply chain consulting context. I start from public shipment pricing data, clean operational noise, engineer delivery and cost features, and forecast demand and price using walk-forward validation. I then score supplier risk from reliability, volatility, and concentration indicators, and use those signals inside a PuLP sourcing optimizer. Finally, I stress-test the recommended order plan with Monte Carlo scenarios and expose the outputs through a Streamlit dashboard. The result is not just a model, but a decision workflow that a procurement or planning team could actually discuss and act on.

## Likely interviewer questions and answers

### Why XGBoost over ARIMA?
XGBoost is useful when demand is influenced by nonlinear effects, sparse history, and engineered lag features. ARIMA is still a strong benchmark for stable univariate patterns, so I compare both instead of assuming one wins.

### Why walk-forward validation?
Random K-fold leaks future information in time series. Walk-forward validation respects temporal order and gives a more realistic estimate of how the model would perform in planning cycles.

### How did you calculate supplier risk?
I combined business-explainable metrics such as late-delivery rate, average delay, price volatility, freight volatility, missing-data rate, and concentration risk into a normalized weighted score from 0 to 100. I added an ML classifier for late-delivery prediction as a second lens.

### Why PuLP optimization?
PuLP is transparent, Python-native, and easy to explain in interviews. It is strong enough to demonstrate constrained sourcing allocation while keeping the logic readable for business stakeholders.

### What are the constraints?
Demand satisfaction, storage capacity, MOQ, risky supplier allocation caps, and ABC-based service levels. Feasible vendor-item pairs are limited to historically observed shipment combinations.

### How would this move to Oracle OCI Data Science?
I would land raw and processed data in OCI Object Storage, develop models in OCI Data Science notebooks or jobs, register trained artifacts in Model Catalog, schedule scoring and optimization with OCI Functions or Data Flow, and serve curated results through Autonomous Database plus a separate Streamlit deployment.

### How would this scale with PySpark?
I would move preprocessing, feature engineering, and vendor-level aggregations to PySpark DataFrames, keep model training on sampled or grouped series where needed, and persist feature tables into a lakehouse or warehouse layer for distributed scoring.

## How this would be deployed on Oracle OCI Data Science
- OCI Object Storage for raw and processed datasets
- OCI Data Science Notebook Sessions for EDA and model development
- OCI Model Catalog for trained forecasting and supplier-risk artifacts
- OCI Functions or OCI Data Flow for scheduled inference and optimization runs
- Oracle Autonomous Database for serving optimized procurement recommendations
- Streamlit dashboard deployed separately as a business-facing UI

## What to capture for the interview
- EDA notebook screenshots:
  - missing-value summary
  - vendor Pareto chart
  - delay distribution
- Forecasting outputs:
  - one actual vs forecast chart
  - model comparison table
- Supplier risk outputs:
  - top risky suppliers
  - risk score distribution
  - feature importance or SHAP chart
- Optimization outputs:
  - recommended order plan table
  - cost savings vs baseline
- Simulation outputs:
  - service-level distribution
  - scenario summary
- Streamlit dashboard screenshots for each page

## Honest interview framing
- This is a public-data prototype, not production client data.
- The optimization parameters such as MOQ, service levels, and storage capacity are configurable assumptions because public Kaggle data does not contain every planning constraint directly.
- The project is built to demonstrate consulting-style problem solving, explainability, and deployment thinking rather than claim real enterprise accuracy.
