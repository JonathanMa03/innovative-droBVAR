import unittest

import numpy as np

from innovcal.api.innovations import fit_innovations
from innovcal.dro.shift_experiments import (
    add_stress_degradation,
    evaluate_shifted_realizations,
    generate_shifted_realizations,
)
from innovcal.evaluation.rolling import evaluate_rolling_forecasts
from innovcal.forecasting.portfolio import (
    portfolio_simple_returns_from_log_returns,
    portfolio_wealth_paths,
)
from innovcal.forecasting.rolling import generate_rolling_var_forecasts
from innovcal.innovations.bootstrap import sample_block_bootstrap_innovations
from innovcal.vector_ar.fit import fit_var_ols


class RobustForecastingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.returns = rng.normal(scale=0.01, size=(100, 3))
        self.residuals = rng.normal(scale=0.01, size=(40, 3))

    def test_exact_portfolio_transform_and_compounding(self):
        log_returns = np.log1p(np.array([[[0.10, -0.05], [0.02, 0.04]]]))
        simple = portfolio_simple_returns_from_log_returns(log_returns, np.array([0.6, 0.4]))
        np.testing.assert_allclose(simple, [[0.04, 0.028]])
        np.testing.assert_allclose(portfolio_wealth_paths(simple), [[1.04, 1.06912]])

    def test_block_bootstrap_preserves_within_block_adjacency(self):
        residuals = np.arange(12, dtype=float)[:, None]
        samples = sample_block_bootstrap_innovations(
            residuals, n_paths=5, horizon=8, block_length=4, seed=2
        )
        self.assertEqual(samples.shape, (5, 8, 1))
        differences = np.diff(samples[:, :, 0], axis=1)
        self.assertTrue(np.all((differences[:, [0, 1, 2, 4, 5, 6]] % 12) == 1))

    def test_volatility_bootstrap_is_finite(self):
        model = fit_innovations(
            self.residuals, method="volatility_bootstrap", volatility_span=10
        )
        samples = model["sample_fn"](6, 5, seed=3)
        self.assertEqual(samples.shape, (6, 5, 3))
        self.assertTrue(np.isfinite(samples).all())

    def test_rolling_forecast_and_evaluation_shapes(self):
        models = {"bootstrap": fit_innovations(self.residuals, method="bootstrap")}
        rolling = generate_rolling_var_forecasts(
            self.returns,
            test_start=70,
            innovation_models=models,
            horizon=5,
            n_paths=10,
            origin_step=5,
            seed=4,
        )
        self.assertEqual(rolling.forecasts["bootstrap"].shape, (6, 10, 5, 3))
        table = evaluate_rolling_forecasts(
            rolling.forecasts, rolling.observations, horizons=(1, 5)
        )
        self.assertEqual(len(table), 2)
        self.assertTrue(np.isfinite(table.select_dtypes("number")).all().all())

    def test_shifted_realization_evaluation_is_common_and_finite(self):
        fitted = fit_var_ols(self.returns[:70], lags=1, include_intercept=True)
        truth_model = fit_innovations(self.residuals, method="block_bootstrap", block_length=4)
        realized, _ = generate_shifted_realizations(
            fitted,
            truth_model,
            self.returns[69:70],
            horizon=5,
            n_realizations=8,
            method="variance_inflation",
            epsilon=0.2,
            seed=5,
        )
        rng = np.random.default_rng(9)
        forecasts = {"a": rng.normal(scale=0.01, size=(20, 5, 3))}
        base = evaluate_shifted_realizations(forecasts, realized, "variance_inflation", 0.0)
        stressed = evaluate_shifted_realizations(forecasts, realized, "variance_inflation", 0.2)
        combined = add_stress_degradation(
            __import__("pandas").concat([base, stressed], ignore_index=True)
        )
        self.assertTrue(np.isfinite(combined.select_dtypes("number")).all().all())


if __name__ == "__main__":
    unittest.main()
