# Model Results Summary

The current run told a fairly clear story. Price was easier to forecast than demand, supplier risk was concentrated in a relatively small set of vendors, and the optimization layer produced the most visible business payoff once those signals were combined.

On the forecasting side, the strongest results came from weekly weighted price rather than weekly demand. For `HIV 1/2, Determine Complete HIV Kit, 100 Tests`, the tree-based model delivered a WAPE of roughly `7.88` on price, while the best demand model for the same series still sat near `94.41`. I do not see that as a modeling failure. Public shipment data often has noisy, irregular item-level quantity behavior, while price moves more smoothly and carries stronger lag structure. In this run, the model was better at learning price discipline than volume timing.

The supplier-risk results were more actionable. The project scored 73 suppliers, with `SCMS from RDC` ranking highest at about `64.2`, followed by `BIO-RAD LABORATORIES (FRANCE)`, `Aurobindo Pharma Limited`, and `CIPLA LIMITED`. Those names did not rise to the top because of one metric alone. They surfaced because delay patterns, volatility, and concentration all stacked in the same direction. That is exactly why I kept the risk view multi-factor instead of treating late delivery as a single-score problem.

The optimization output was the most striking result numerically. The optimized plan came out at about `242.4M`, compared with `578.5M` for the cheapest-supplier-only baseline and `613.0M` for the historical-allocation baseline. I would describe those as modeled scenario savings, not forecasted realized savings. The value is in showing that once landed cost, risk, and allocation constraints are handled together, the procurement decision changes materially.

The simulation layer added an important correction to the deterministic story. Base-case average service landed around `0.909`, and the supplier-delay scenario stayed close to that level. The high-demand scenario was the real stress point, pulling service down to roughly `0.787`. That result matters more than any single forecast metric because it points to where the planning system is actually vulnerable. In this setup, the bigger risk is demand shock rather than the modeled delay shock.

Taken together, the run supports a simple conclusion: the forecast is useful, but the real value appears when forecast, supplier reliability, and sourcing logic are tied together and then tested under uncertainty.
