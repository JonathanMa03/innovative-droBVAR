# innovcal

`innovcal` is a research framework for calibrated and robust multivariate financial forecasting. It separates predictable cross-asset dynamics from the joint uncertainty that remains after forecasting, then evaluates how credible that uncertainty is and how it deteriorates under distributional change.

**Author:** Jonathan Ma

## Research aim

The central question is:

> How does the choice of joint residual model affect the calibration and robustness of multivariate autoregressive forecasts under changes in volatility, dependence, and tail behavior?

A supporting methodological question is:

> Under what residual structures and distributional shifts does diffusion-based innovation modeling provide meaningful improvements over Gaussian, Student-t, and empirical alternatives?

The project does not assume that diffusion is always the best innovation model. Diffusion-Innovation VAR (DI-VAR) is evaluated as an unconditional generative baseline. Its extension, Conditional Diffusion-Innovation VAR (CDI-VAR), conditions standardized joint innovations on recent residual history and a causal volatility state, then applies regularized adaptive calibration to the completed VAR forecast distribution.

## Framework

The methodology has four layers:

1. **Predictable dynamics:** A vector autoregression (VAR) models lagged interactions among a small set of related financial assets.
2. **Joint residual uncertainty:** Competing innovation models learn the multivariate distribution of rolling VAR forecast errors.
3. **Calibration:** Probabilistic forecasts are assessed for marginal, joint, and tail reliability as well as sharpness. CDI-VAR additionally updates bounded forecast-scale corrections using only outcomes observable at each rolling origin.
4. **Stress testing:** Innovation distributions are perturbed to measure forecast and portfolio-risk degradation under plausible distributional shifts.

For a return vector $y_t \in \mathbb{R}^K$,

$$
y_t = c + \sum_{\ell=1}^{p} A_\ell y_{t-\ell} + u_t,
$$

where the VAR estimates the conditional mean and a joint innovation model estimates the distribution of $u_t$. Future innovation samples are propagated recursively through the fitted VAR to generate probabilistic return, price, and portfolio paths.

## Core workflow

1. Collect and align adjusted prices for a small set of economically related assets.
2. Transform prices into stationary series, using log returns by default.
3. Split observations chronologically into training, calibration, and test periods.
4. Fit a VAR to the training data.
5. Produce rolling one-step-ahead errors without using future information.
6. Fit competing joint innovation models to calibration residuals.
7. Sample joint innovations and recursively generate multistep forecast paths.
8. Evaluate marginal, multivariate, and tail calibration on untouched test periods.
9. Apply volatility, dependence, downside-tail, outlier, and Wasserstein-based stresses.
10. Measure degradation in forecast calibration and portfolio-risk estimates.

## Models under comparison

The conditional-mean model is held fixed wherever possible so differences can be attributed to the innovation distribution.

| Model | Joint innovation specification | Role |
|---|---|---|
| Gaussian-VAR | Multivariate Gaussian | Conventional benchmark |
| Student-t-VAR | Multivariate Student-t | Parametric heavy-tail benchmark |
| Bootstrap-VAR | Joint or block residual resampling | Empirical nonparametric benchmark |
| DI-VAR | Diffusion-generated residual vectors | Flexible generative candidate |
| CDI-VAR | Volatility-aware conditional diffusion with adaptive calibration | Proposed conditional and calibrated model |

Residual vectors are modeled jointly to preserve contemporaneous cross-asset dependence. A block bootstrap can additionally preserve short-range temporal dependence.

## CDI-VAR specification

CDI-VAR preserves a deliberately interpretable decomposition:

1. A common VAR estimates the conditional mean.
2. A causal EWMA recursion estimates marginal residual scale.
3. A conditional diffusion model generates standardized joint shocks given recent standardized residuals and current log volatility.
4. Raw shocks update the latent residual and volatility state.
5. A separate calibration layer rescales completed VAR forecast deviations by lead time.

For residual component $u_{t,j}$, the causal volatility state is

$$
v_{t+1,j}=\lambda v_{t,j}+(1-\lambda)u_{t,j}^{2},
\qquad
z_{t,j}=\frac{u_{t,j}}{\sqrt{v_{t,j}}},
$$

where $v_t$ is known before observing $u_t$. The diffusion context concatenates the most recent standardized residual vectors with normalized log volatility. During simulation, calibration never feeds back into this recursion: the state is updated using raw conditional draws, and calibration is applied only after those draws have been propagated through the VAR.

Calibration uses genuine rolling VAR forecast outcomes from a chronologically reserved block. Raw scale estimates are shrunk toward one and bounded before deployment. At test origin $o$, adaptive updates use only forecast outcomes whose realization dates are no later than $o$. The current research configuration is:

| Parameter | Default | Interpretation |
|---|---:|---|
| VAR lags | 1 | Conditional-mean lag order |
| Conditional residual lags | 5 | Standardized residual vectors supplied as diffusion context |
| EWMA span | 60 | Causal marginal-volatility memory |
| Diffusion steps | 50 in experiments | Reverse-process discretization |
| Hidden dimension | 128 | Conditional denoising-network width |
| Training epochs | 300 maximum | Early stopping selects the checkpoint |
| Validation fraction | 0.15 | Chronological checkpoint-selection block |
| Calibration fraction | 0.15 | Later chronological calibration block |
| Calibration anchors | 1, 5, 20 days | Explicit forecast leads calibrated |
| Calibration paths | 32 | Monte Carlo paths per calibration origin |
| Shrinkage | 0.5 | Pulls raw multipliers halfway toward one |
| Multiplier bounds | $[0.8,1.25]$ | Prevents unstable widening or sharpening |
| Adaptive window | 12 origins | Most recent available forecast cases used online |

These are frozen experimental settings, not universally optimal financial constants. Any later tuning must occur inside a new training/calibration design rather than against reported test results.

## Evaluation

Forecast quality is evaluated through:

- prediction-interval coverage, width, and interval score,
- probability integral transform diagnostics,
- expected calibration error,
- Continuous Ranked Probability Score,
- Energy Score and other multivariate scoring rules,
- joint downside-event frequencies,
- Value at Risk exceedances and Expected Shortfall,
- portfolio loss and drawdown distributions,
- computational cost and stability across repeated runs.

Calibration and sharpness are reported together: wide intervals can achieve coverage without producing useful forecasts. Asset-level diagnostics are supplemented by joint and portfolio-level measures because marginal calibration can conceal misspecified dependence.

## Robustness and stress testing

Robustness is treated as an evaluated property, not an automatic consequence of flexibility. The framework studies forecast sensitivity under:

- volatility inflation,
- stronger common dependence and downside correlation,
- heavier or asymmetric negative tails,
- isolated asset shocks and outlier contamination,
- historical market regimes,
- distributions within controlled Wasserstein neighborhoods of the fitted residual law.

The stress layer asks how quickly calibration, tail coverage, and portfolio-risk estimates deteriorate as the innovation distribution moves away from its reference distribution.

## Empirical and simulation studies

The primary application uses a small multivariate system of related financial assets. Daily or weekly adjusted prices are converted to log returns, modeled through rolling forecast origins, and evaluated on chronologically later observations.

Controlled simulations remain part of the project because they reveal when an innovation model succeeds or fails under known distributions. Current regimes include Gaussian, Student-t, mixture, and heteroskedastic innovations. Repeated simulations provide identification; the financial case study provides external validity.

## Interpretation

The framework preserves interpretable boundaries:

- VAR coefficients describe lagged cross-asset relationships.
- Residual diagnostics describe behavior left unexplained by the VAR.
- Innovation-model comparisons identify the complexity required to represent that uncertainty.
- Calibration diagnostics determine whether forecast probabilities are credible.
- Stress-response curves show where reliability breaks down.
- Portfolio metrics translate statistical failures into financial consequences.

DI-VAR is therefore a candidate innovation specification rather than the identity of the project. A simpler model outperforming diffusion is an informative result, especially in low-dimensional or data-limited settings.

## Repository organization

```text
src/innovcal/
├── api/            # high-level workflow functions
├── data/           # simulation, loading, and preprocessing
├── vector_ar/      # VAR fitting, diagnostics, stability, and forecasting
├── di_var/         # high-level DI-VAR workflow and rolling residuals
├── cdi_var/        # conditional diffusion, EWMA state, and adaptive calibration
├── innovations/    # Gaussian, bootstrap, and Student-t residual models
├── diffusion/      # diffusion training and sampling
├── forecasting/    # recursive trajectories and Monte Carlo forecasts
├── calibration/    # coverage, PIT, ECE, and reliability
├── evaluation/     # proper scoring rules and summary metrics
├── dro/            # perturbations, Wasserstein tools, and stress tests
├── experiments/    # configuration, orchestration, and artifacts
└── plots/          # visualization utilities
```

## Immediate research priorities

1. Freeze the current CDI-VAR specification before further test-period analysis.
2. Complete CDI-VAR ablations for conditioning, volatility, and calibration.
3. Add nested walk-forward evaluation across historical regimes.
4. Quantify paired score uncertainty with dependence-aware confidence intervals.
5. Extend controlled simulations with persistent volatility, residual dependence, and regime shifts.
6. Report computational cost, calibration trajectories, and failure cases.

## Notebook workflow

The notebooks are intentionally thin and should be run in order:

1. `00_environment_and_tests.ipynb`
2. `01a_financial_data.ipynb` (retained demo; not part of the empirical run)
3. `01b_real_world_data.ipynb`
4. `02_var_and_residuals.ipynb`
5. `03_classical_innovations.ipynb`
6. `04_di_var.ipynb`
7. `04a_cdi_var.ipynb`
8. `05_calibration_and_portfolio.ipynb`
9. `06_stress_tests.ipynb`
10. `07_simulation_validation.ipynb`
11. `08_results.ipynb`
12. `09_repeated_results.ipynb`

Intermediate artifacts are written to `results/notebook_cache/`. The market-data notebook downloads AAPL, JPM, XOM, and WMT adjusted closes for 2007--2025, saves the cleaned price panel to `data/processed/market_prices.csv`, and saves the canonical log-return panel to `data/processed/financial_returns.csv`.

## Future direction

The framework supports later work on distributional shift. Future extensions can monitor rolling residual distributions, detect changes in covariance or tail dependence, distinguish innovation shift from changing conditional-mean dynamics, and adapt the innovation model or robustness radius online. A central implemented extension already replaces the unconditional innovation law $p(u)$ with the CDI-VAR law $p_{\theta}(z_{t+1}\mid z_{t-L+1:t},v_{t+1})$. Future work will examine richer state variables and formal shift detection.

## Status

This repository is under active research and development. It currently provides the real-data pipeline, leakage-aware rolling residual construction, classical innovation models, DI-VAR, CDI-VAR, rolling-origin evaluation, exact portfolio transformations, repeated-seed experiments, and shifted-realization stress tests. Ablation, nested regime evaluation, statistical comparison, and final thesis synthesis remain to be completed.
