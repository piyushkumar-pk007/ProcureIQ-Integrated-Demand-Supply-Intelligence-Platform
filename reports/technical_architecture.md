# Technical Architecture

## Pipeline layers
1. `src/data_ingestion.py`
   Downloads the Kaggle dataset if credentials are available or reads a manually downloaded CSV from `data/raw/`.
2. `src/preprocessing.py`
   Standardizes columns, parses dates, creates missing flags, converts numerics, and saves a cleaned dataset.
3. `src/feature_engineering.py`
   Adds delay, lead time, landed-cost, vendor performance, and weekly demand/price features.
4. `src/forecasting.py`
   Runs walk-forward time-series validation with baseline, SARIMA, optional Prophet, and ML models.
5. `src/supplier_risk.py`
   Produces explainable supplier risk scoring and an ML late-delivery risk model.
6. `src/optimization.py`
   Solves a procurement allocation problem using PuLP under service-level, capacity, MOQ, and risk constraints.
7. `src/simulation.py`
   Runs Monte Carlo scenario analysis across demand, delay, and inflation stress cases.
8. `app/streamlit_app.py`
   Serves the business dashboard for executive, analytical, and operational views.

## Storage pattern
- `data/raw/`: source CSV files
- `data/processed/`: cleaned and engineered analytical datasets
- `data/outputs/`: model results, optimization outputs, simulation outputs, and plots

## Cloud-ready migration path
- Replace local CSV storage with OCI Object Storage.
- Run notebooks and scripts inside OCI Data Science notebook sessions or jobs.
- Register trained forecasting and risk models in OCI Model Catalog.
- Trigger scheduled scoring and optimization through OCI Functions or Data Flow.
- Persist curated outputs to Oracle Autonomous Database for downstream reporting.
- Deploy the Streamlit app separately as a lightweight presentation layer.

