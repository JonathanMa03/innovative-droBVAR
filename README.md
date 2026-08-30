# innovcal

`innovcal` is a research framework for calibrated and robust multivariate financial forecasting. It separates predictable cross-asset dynamics from the joint uncertainty that remains after forecasting, then evaluates how credible that uncertainty is and how it deteriorates under distributional change.

**Author:** Jonathan Ma

## Research aim

The central question is:

> How does the choice of joint residual model affect the calibration and robustness of multivariate autoregressive forecasts under changes in volatility, dependence, and tail behavior?

A supporting methodological question is:

> Under what residual structures and distributional shifts does diffusion-based innovation modeling provide meaningful improvements over Gaussian, Student-t, and empirical alternatives?

The project does not assume that diffusion is always the best innovation model. Diffusion-Innovation VAR (DI-VAR) is evaluated as one candidate inside a broader and interpretable forecasting framework.

## Framework

The methodology has four layers:

1. **Predictable dynamics:** A vector autoregression (VAR) models lagged interactions among a small set of related financial assets.
2. **Joint residual uncertainty:** Competing innovation models learn the multivariate distribution of rolling VAR forecast errors.
3. **Calibration:** Probabilistic forecasts are assessed for marginal, joint, and tail reliability as well as sharpness.
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

Residual vectors are modeled jointly to preserve contemporaneous cross-asset dependence. A block bootstrap can additionally preserve short-range temporal dependence.

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

1. Add a reproducible market-data pipeline and define the asset-selection rationale.
2. Replace purely in-sample residual fitting with rolling out-of-sample residual construction.
3. Establish a strict training/calibration/test protocol.
4. Compare innovation models using identical VAR fits and forecast origins.
5. Add multivariate and portfolio-level calibration diagnostics.
6. Complete stress experiments for volatility, dependence, and downside-tail shifts.
7. Repeat simulation and empirical evaluations across seeds and time windows.
8. Report uncertainty, computational cost, failure cases, and simple-model baselines.

## Notebook workflow

The notebooks are intentionally thin and should be run in order:

1. `00_environment_and_tests.ipynb`
2. `01_financial_data.ipynb`
3. `02_var_and_residuals.ipynb`
4. `03_classical_innovations.ipynb`
5. `04_di_var.ipynb`
6. `05_calibration_and_portfolio.ipynb`
7. `06_stress_tests.ipynb`
8. `07_simulation_validation.ipynb`
9. `08_results.ipynb`

Intermediate artifacts are written to `results/notebook_cache/`. If no real adjusted-price CSV is present at `data/raw/market_prices.csv`, the data notebook uses a clearly labeled deterministic demo dataset so the complete workflow can be tested. Demo results are not empirical evidence.

## Future direction

The framework supports later work on distributional shift. Future extensions can monitor rolling residual distributions, detect changes in covariance or tail dependence, distinguish innovation shift from changing conditional-mean dynamics, and adapt the innovation model or robustness radius online. A further extension would replace the unconditional innovation law $p(u)$ with a market-state-dependent law $p(u_{t+1}\mid\mathcal{F}_t)$.

## Status

This repository is under active research and development. The existing implementation provides simulation, VAR and probabilistic RNN components, classical and diffusion innovation models, recursive forecasting, calibration metrics, and initial stress-testing utilities. The financial data study, repeated rolling evaluation, and final robustness analysis remain to be completed.
