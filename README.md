# ProcureIQ — Integrated Demand-Supply Intelligence Platform

ProcureIQ is a supply chain analytics project I built around a procurement planning question that shows up all the time in practice: demand is moving, suppliers are not equally reliable, freight behavior is messy, and someone still has to decide where to place orders. I used a public shipment-pricing dataset to connect those pieces into one workflow: clean the data, forecast near-term demand and price, score supplier risk, optimize allocations, and then stress-test the plan under uncertainty.

I built it because procurement analytics often stops too early. A lot of portfolio work ends with a forecast chart or a vendor scorecard. That is useful, but it leaves out the harder part, which is turning those signals into an actual sourcing decision. I wanted this repo to show that the modeling matters only if it changes what the planner does next.

**This is a Kaggle-based demonstration inspired by real supply-chain procurement and forecasting problems.**

## The data and what it cannot tell us

The project uses Kaggle’s `apoorvwatsky/supply-chain-shipment-pricing-data`. The dataset has enough operational detail to say something useful about supplier behavior: purchase order dates, scheduled and actual delivery dates, quantities, item values, pack and unit prices, freight, insurance, shipping mode, vendor, country, and product descriptors. That made it possible to build delay features, landed-cost features, vendor-level consistency measures, and weekly planning snapshots.

The limitation is just as important as the signal. This is public transactional data, not a live ERP or procurement system. It does not come with negotiated MOQ rules, storage policies, sourcing contracts, or a clean vendor-item master. I modeled those assumptions explicitly in code and kept them visible. That makes the project more honest and, in my view, more useful.

## How I approached it

The pipeline starts with ingestion because even small public datasets tend to be messier than they look. The loader supports Kaggle API download when credentials are available and manual CSV placement when they are not. From there I standardized columns, parsed the operational dates, converted numeric fields that arrived as mixed text, and created missing-value flags before any imputation. One design choice I cared about was avoiding silent row loss. The current run kept all 10,324 records through cleaning, which matters because late-delivery rates and vendor price behavior can shift quickly if rows disappear without explanation.

Once the base table was stable, I built features that make procurement data behave more like a planning dataset. Delivery delay, verification delay, total landed cost, freight per kilogram, value per unit, weekly demand, weekly weighted price, vendor delay history, vendor late rate, and vendor price volatility all feed into the next layers. At that point the project branches in three directions that eventually come back together.

The first branch is forecasting. I compared seasonal naive, moving average, SARIMA, Prophet when available, and a lag-based tree model with XGBoost as the preferred option. The model choice was empirical rather than ideological. I wanted a classical baseline, a simple benchmark a planner can explain in a meeting, and a nonlinear model that can pick up patterns in noisy weekly series. Validation is walk-forward because random splits would overstate performance for time-dependent data.

The second branch is supplier risk. I kept both a rule-based score and an ML view of late-delivery risk. The weighted score works better when you need something transparent enough for procurement or operations to challenge. The classifier works better when behavior is nonlinear and history interacts in ways a hand-built score misses. I treated those as complementary rather than competing views.

The third branch is procurement optimization. I used PuLP to convert demand, price, supplier risk, MOQ assumptions, storage limits, and service-level targets into a sourcing recommendation. The optimizer only allows supplier-item combinations that appeared historically. That restriction is simple, but it prevents the classic prototype problem where the math recommends a pairing that never existed operationally.

I added Monte Carlo simulation after optimization because deterministic plans usually look cleaner than they behave. A plan that meets service targets at average demand can still fail badly when demand jumps or delays cluster. The simulator lets the optimized plan absorb demand shocks, supplier delays, and cost inflation before I trust the recommendation.

## What came out of the current run

The forecasting layer was strongest on price behavior for high-history HRDT series. For `HIV 1/2, Determine Complete HIV Kit, 100 Tests`, the weighted price forecast from the tree model finished with a WAPE of about `7.88`, while the corresponding demand series remained much harder, with the best WAPE still around `94.41`. That is a realistic outcome for procurement data: price often moves more smoothly than sparse item-level demand.

The supplier-risk layer scored 73 suppliers. The highest-risk names in the current run were `SCMS from RDC` at roughly `64.2`, followed by `BIO-RAD LABORATORIES (FRANCE)`, `Aurobindo Pharma Limited`, and `CIPLA LIMITED`, all in the `50` range. Those scores were driven by combinations of delay behavior, volatility, and concentration, not just by on-time rate alone.

The optimizer produced the clearest business result. The optimized plan came out at about `242.4M` in modeled total cost, compared with `578.5M` for a cheapest-supplier-only baseline and `613.0M` for a historical-allocation baseline. I would never present those figures as guaranteed savings, because they are model outputs built on public data and explicit assumptions, but they are still useful. They show that once supplier risk and landed-cost structure are modeled together, the cheapest visible option is not automatically the best one.

The simulation results helped separate robust improvements from fragile ones. In the base case the average service level was about `0.909`. Under supplier delay shock it held close to that level, but under the high-demand scenario it dropped to about `0.787`. That was the most useful finding from the simulation layer. In this setup, demand shock is a bigger threat to service than the delay shock I modeled, which suggests the next improvement should probably focus on demand sensing, buffers, or service policy before adding even more supplier logic.

## How to run it

I kept the local workflow simple because the point of the repo is the analysis, not setup gymnastics.

```powershell
cd "f:\Portfolio\Supply Chain - DS Project\procureiq-demand-supply-intelligence"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you want Kaggle download, copy `.env.example` to `.env` and add `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `KAGGLE_DATASET=apoorvwatsky/supply-chain-shipment-pricing-data`. If you would rather skip the API, place the raw CSV in `data/raw/`.

From there I usually run the whole project through the wrapper script:

```powershell
python run_project.py
```

If you want the steps individually, the modules can be run directly:

```powershell
python -m src.data_ingestion
python -m src.preprocessing
python -m src.feature_engineering
python -m src.forecasting
python -m src.supplier_risk
python -m src.optimization
python -m src.simulation
```

The dashboard runs separately:

```powershell
python -m streamlit run app\streamlit_app.py
```

And the tests are standard:

```powershell
python -m pytest
```

## What I would say are the real limitations

The first limitation is the data itself. This is shipment history from Kaggle, not a production procurement stack. Supplier contracts, real MOQ rules, business calendars, and sourcing policies are approximated rather than observed directly. The second limitation is time-series depth. Some item-level demand series are simply too thin to support elegant forecasting, so the code has to fall back to more aggregated views instead of pretending the history is richer than it is.

The third limitation is that optimization outputs are only as good as the assumptions around them. I exposed those assumptions in `src/config.py`, but they are still assumptions. A real deployment would pull them from planning policy, procurement contracts, and warehouse operations. Finally, the savings and service-level outputs are scenario estimates, not realized business outcomes. I think saying that plainly makes the project more credible, not less.

## Closing thought

What I like about ProcureIQ is that it treats procurement as a connected decision problem. The forecast is not the endpoint. The supplier score is not the endpoint either. They only become valuable when they roll into an order recommendation and that recommendation survives a harder look under uncertainty.
