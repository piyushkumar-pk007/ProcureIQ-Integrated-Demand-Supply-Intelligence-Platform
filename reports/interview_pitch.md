# ProcureIQ Interview Pitch

## 30-second pitch
I built ProcureIQ, a Kaggle-based supply chain intelligence prototype that combines demand and price forecasting, supplier risk scoring, and procurement optimization. It forecasts weekly demand and unit price, identifies risky suppliers using delivery and price volatility signals, and recommends optimized supplier allocation using PuLP. The system demonstrates how data science can support S&OP, procurement, and inventory decisions.

## 2-minute pitch
ProcureIQ was designed as an end-to-end decision intelligence prototype for procurement and supply-chain teams. I used the public Kaggle dataset "Supply Chain Shipment Pricing Data" to simulate a consulting-style use case where a company wants better visibility into demand, supplier performance, and sourcing decisions.

The first layer is data engineering and forecasting. I built a modular pipeline to ingest the raw CSV from Kaggle or a manual download, standardize messy operational fields, parse dates, create data quality flags, and engineer time-series features. From there, I forecast weekly demand and weighted average unit price for the most relevant item or product-group series using seasonal baselines, SARIMA, and an ML regressor with lag features under walk-forward validation.

The second layer is supplier risk. I created a transparent rule-based supplier risk score from late delivery rate, delay volatility, price volatility, freight variability, missing data rate, and concentration signals. I paired that with an ML classifier that predicts late delivery at the shipment level, so the project demonstrates both explainability for business users and predictive modeling for operations teams.

The third layer is procurement optimization. I used PuLP to allocate forecasted demand across historically feasible vendor-item combinations while minimizing purchase cost, freight, holding cost, and supplier risk penalties under service-level, MOQ, storage-capacity, and risky-supplier allocation constraints. I then stress-tested the optimized plan with Monte Carlo simulation across demand shocks, supplier delay shocks, and cost inflation scenarios.

The final output is a Streamlit dashboard that packages the pipeline into an interview-ready business story: what demand is coming, which suppliers are risky, how sourcing should be adjusted, and what the expected savings and service-level impact could be.

