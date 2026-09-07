# CA-RNN

This repository develops the **Calibration-Aware Recurrent Neural Network (CA-RNN)** for multivariate probabilistic time-series forecasting. CA-RNN augments likelihood-based recurrent training with differentiable penalties that target calibration of the predictive distribution.

The project follows the methodological and experimental template of *Calibration-Aware Bayesian Learning*, adapted from classification to continuous sequential prediction. The four primary models are:

| Reference-paper role | Sequential model |
|---|---|
| Frequentist neural network | RNN |
| Bayesian neural network | Bayesian RNN (BRNN) |
| Calibration-aware neural network | CA-RNN |
| Calibration-aware Bayesian network | CA-BRNN |

The adaptation is not a direct reuse of classification ECE. Continuous multivariate forecasts require a calibration definition for joint predictive distributions and an additional check that forecast errors are not predictable over time.

## Research questions

The primary question is:

> Does direct calibration-aware training improve multivariate probabilistic RNN forecasts, and does that improvement persist under temporal and distributional shift without an unacceptable loss of sharpness or predictive accuracy?

The experiments address four supporting questions:

1. How does the calibration weight affect calibration, proper predictive scores, sharpness, and point accuracy?
2. Does calibration-aware Bayesian learning behave differently from its frequentist counterpart in sequential data?
3. Does explicitly penalizing temporal dependence in probability integral transforms improve sequential forecast adequacy?
4. How rapidly do the four models deteriorate under volatility, dependence, tail, nonlinear, and structural changes?

## Predictive model

For a multivariate series $Y_t\in\mathbb R^d$, a GRU encodes the available history:

$$
h_t=\operatorname{GRU}_\theta(h_{t-1},Y_t).
$$

The predictive head returns a location vector and lower-triangular scale matrix:

$$
Y_{t+1}\mid\mathcal F_t
\sim
\mathcal N_d\left(\mu_\theta(h_t),L_\theta(h_t)L_\theta(h_t)^\top\right).
$$

Positive diagonal transformations of $L_\theta$ guarantee a positive-definite covariance matrix. The initial study uses the joint Gaussian head because every projected predictive CDF is analytic and differentiable. Heavy-tailed heads are reserved for a prespecified extension.

The ordinary RNN minimizes the proper negative log-likelihood objective

$$
\mathcal L_{\mathrm{NLL}}
=-rac{1}{T}\sum_t
\log p_\theta(Y_{t+1}\mid\mathcal F_t).
$$

## Projected multivariate calibration

Correct univariate marginals do not guarantee a correct joint forecast. For fixed unit projection vectors $a_1,\ldots,a_R$, CA-RNN computes projected probability integral transforms

$$
U_{t,r}=F_{\theta,t,a_r}(a_r^\top Y_t).
$$

Under a correctly specified continuous forecast, $U_{t,r}$ is uniformly distributed for every projection. The projection set contains asset coordinates, an equal-weight portfolio, and reproducibly generated random directions.

The calibration objective is a smooth Cramér--von Mises discrepancy. For grid values $q_g\in(0,1)$,

$$
\mathcal L_{\mathrm{cal}}
=
\frac{1}{GR}
\sum_{g=1}^G\sum_{r=1}^R
\left[
\frac{1}{T}\sum_t
\sigma\left(\frac{q_g-U_{t,r}}{\tau}\right)-q_g
\right]^2,
$$

where $\sigma$ is the logistic function and $\tau>0$ controls the differentiable approximation to the empirical CDF.

## Sequential calibration

Uniform PIT values can still contain temporal structure. The sequential extension penalizes their lagged covariance:

$$
\mathcal L_{\mathrm{seq}}
=
\frac{1}{RK}
\sum_{r=1}^R\sum_{k=1}^K
\left[
\frac{1}{T-k}\sum_{t=k+1}^T
(U_{t,r}-1/2)(U_{t-k,r}-1/2)
\right]^2.
$$

The full frequentist objective is

$$
\boxed{
\mathcal L_{\mathrm{CA\text{-}RNN}}
=
\mathcal L_{\mathrm{NLL}}
+\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}
+\lambda_{\mathrm{seq}}\mathcal L_{\mathrm{seq}}.
}
$$

The main CA-RNN comparison initially sets $\lambda_{\mathrm{seq}}=0$ to match the reference paper's calibration-aware design. The sequential term is introduced as a separate ablation and extension.

## Bayesian formulation

The BRNN places a mean-field Gaussian variational distribution $q_\phi(\theta)$ over the recurrent and predictive-head parameters. Its calibration-aware variational objective is

$$
\boxed{
\mathcal L_{\mathrm{CA\text{-}BRNN}}(\phi)
=
\mathbb E_{\theta\sim q_\phi}
\left[
\mathcal L_{\mathrm{NLL}}(\theta)
+\lambda_{\mathrm{cal}}\mathcal L_{\mathrm{cal}}(\theta)
+\lambda_{\mathrm{seq}}\mathcal L_{\mathrm{seq}}(\theta)
\right]
+\frac{\beta}{N}
\operatorname{KL}\!\left(q_\phi(\theta)\Vert p(\theta)\right).
}
$$

The implementation samples one set of recurrent weights per sequence evaluation, uses the reparameterization trick, and includes the KL term in both ordinary BRNN and CA-BRNN training. The KL weight $\beta$ must be selected on validation data.

## Experimental workflow

The experiment sequence deliberately parallels the calibration-aware Bayesian learning study:

1. Establish controlled multivariate sequential data-generating processes.
2. Compare RNN, BRNN, CA-RNN, and CA-BRNN using matched architectures and splits.
3. Sweep $\lambda_{\mathrm{cal}}$ and plot calibration against predictive performance.
4. Compare nominal and empirical probabilities through PIT and coverage reliability plots.
5. Ablate the projected calibration and sequential PIT penalties.
6. Evaluate the four models under progressively stronger distribution shifts.
7. Apply the frozen workflow to four financial return series.

All transformations and splits are chronological. Model selection uses validation data only. Final evaluations use untouched test observations and fresh simulation seeds. Calibration improvements are always reported with proper scores and interval widths to expose trivial widening.

## Evaluation metrics

The primary measures are:

- projected-PIT calibration error and Kolmogorov--Smirnov statistics;
- lagged PIT dependence;
- nominal interval coverage and reliability curves;
- interval width and interval score;
- multivariate Energy Score;
- root mean squared error;
- numerical failures, training time, and seed sensitivity.

Coordinate PITs assess marginal distributions. Portfolio and random projections probe cross-series dependence. No individual diagnostic is treated as proof of joint calibration.

## Distribution shift

Controlled experiments vary one mechanism at a time where possible:

- marginal volatility;
- contemporaneous dependence;
- heavy and asymmetric tails;
- conditional heteroskedasticity;
- nonlinear conditional dynamics;
- gradual drift and abrupt structural breaks.

Shift experiments measure degradation curves rather than a single stressed endpoint. Stress results do not participate in hyperparameter selection.

## Financial application

The empirical application uses daily adjusted prices for AAPL, JPM, XOM, and WMT beginning in 2007. The cleaned series are daily log returns. These assets provide a small but economically varied multivariate system spanning major market regimes.

Financial data are stored locally under `data/processed/` and are ignored by Git. Notebook 06 can use an existing clean panel or download a new one through the optional `yfinance` dependency.

## Repository structure

```text
src/innovcal/
├── ca_rnn/       # models, variational layers, losses, training, prediction
├── data/         # chronological windows, market data, controlled DGPs
├── evaluation/   # calibration, proper scores, coverage, PIT diagnostics
└── experiments/  # four-model comparisons, sweeps, and shift curves

notebooks/
├── 00_environment_and_tests.ipynb
├── 01_data_and_dgps.ipynb
├── 02_four_model_comparison.ipynb
├── 03_calibration_weight_sweep.ipynb
├── 04_sequential_calibration_ablation.ipynb
├── 05_distribution_shift.ipynb
├── 06_financial_application.ipynb
└── 07_results_summary.ipynb
```

Notebooks contain experiment configuration, execution, and interpretation. Reusable modeling and evaluation logic belongs in `src/innovcal`.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,market]'
python -m ipykernel install --user --name innovcal --display-name "Python (innovcal)"
```

Run the tests with:

```bash
python -m pytest -q
```

Then open `notebooks/00_environment_and_tests.ipynb` and select the **Python (innovcal)** kernel.

## Current status

The CA-RNN foundation is implemented. It includes deterministic and mean-field Bayesian GRUs, full-covariance Gaussian predictive heads, projected-PIT calibration, sequential PIT regularization, leakage-safe data utilities, controlled DGPs, sample-based evaluation, matched four-model experiments, and unit tests.

The notebook settings are initial development settings, not frozen thesis specifications. The next step is to run smoke experiments, inspect numerical behavior, and preregister the confirmatory hyperparameter grids and seeds before producing thesis results.
