# Frozen confirmatory replication protocol

Protocol version: 1.0  
Frozen before inspecting the confirmatory asset panel or its model results.

## Aim and primary contrast

The confirmatory experiment tests whether the previously developed sequential
CA-RNN improves calibration relative to a likelihood-trained RNN on a disjoint
four-asset panel. The sole primary model contrast is **sequential CA-RNN minus
RNN**. Conventional baselines contextualize performance but are secondary.

## Data and split

- Assets, fixed in this order: MSFT, BAC, CVX, COST.
- Daily adjusted closing prices from 2007-01-01 through the last trading day
  strictly before 2026-09-01.
- Align dates by complete cases, then compute log returns.
- No asset substitutions are allowed after results are inspected. A download or
  delisting failure is reported as a protocol failure.
- Use chronological 60% training, 20% validation, and 20% test partitions.
- Fit componentwise standardization on training observations only.
- Use the final 20 observations of the preceding partition as validation/test
  context so that every target has a strictly historical context.

## Frozen neural specification

- One-layer GRU, hidden dimension 32, history length 20.
- Full-covariance Gaussian predictive head.
- RNN objective: NLL only.
- Sequential CA-RNN objective: NLL + 100 calibration loss + 2000 sequential loss.
- PIT grid: the implementation's fixed 19-point grid.
- Projection matrix: coordinates, equal-weight direction, and fixed random
  directions generated with seed 606.
- Maximum 200 epochs, batch size 64, patience 25, validation-NLL checkpointing.
- Training seeds: 611, 619, 631, 641, 647, 653, 661, 673, 683, 691.
- Forecast seed 20606 and 500 predictive draws.
- No tuning or architectural changes may be made using confirmatory test results.

## Conventional baselines

1. Centered historical joint bootstrap.
2. Gaussian VAR with lag selected from {1, 5, 10, 20} by validation one-step MSE.
3. The same VAR mean with marginal Gaussian-QMLE GARCH(1,1) scales and a joint
   training standardized-residual bootstrap.

Baseline parameters are fitted using training data. Validation data select only
the VAR lag. Every test forecast is one-step-ahead and uses realized information
only through the preceding observation.

## Outcomes and inference

Primary outcomes, jointly interpreted without selecting a single favorable
metric, are projected-PIT calibration error, 90% marginal coverage error,
Energy Score, and marginal interval score. RMSE, interval width, projected PIT
mean absolute autocorrelation, and coordinate-only PIT diagnostics are secondary.

For RNN versus sequential CA-RNN, use a paired hierarchical bootstrap over the
ten optimization seeds and 20-trading-day moving blocks, with 2,000 bootstrap
draws and seed 1208. A result is called resolved only when the two-sided 95%
interval excludes zero. Report effect estimates and intervals regardless of
direction. Comparisons with conventional baselines repeat each baseline forecast
array across neural seeds; their intervals therefore condition on the fitted
baseline and represent test-origin plus neural optimization uncertainty.

## Failure and deviation reporting

Report all failed neural fits, non-finite forecasts, and marginal GARCH optimizer
failures. The GARCH implementation may use its documented stable fallback, but
the affected margin must be identified. Any deviation from this file must be
dated, justified, and labeled exploratory; confirmatory results must not be
silently replaced.
