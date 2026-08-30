import unittest

import numpy as np

from innovcal.api.innovations import fit_innovations, sample_innovations
from innovcal.cdi_var.calibration import (
    apply_forecast_calibration,
    fit_adaptive_multipliers,
    fit_scale_multiplier,
    regularize_anchor_multipliers,
)
from innovcal.cdi_var.model import CDIVAR, CDIVARConfig
from innovcal.cdi_var.diffusion import ConditionalResidualDenoiser
from innovcal.cdi_var.innovation import _sample_recursive
from innovcal.diffusion.schedules import make_ddpm_schedule


class CDIVARTests(unittest.TestCase):
    def test_conditional_innovation_fit_and_sample(self):
        rng = np.random.default_rng(4)
        residuals = rng.normal(scale=0.01, size=(90, 3))
        model = fit_innovations(
            residuals,
            method="cdi_var",
            context_lags=3,
            volatility_span=10,
            validation_fraction=0.15,
            calibration_fraction=0.15,
            calibration_horizons=(1, 2),
            calibration_paths=4,
            timesteps=3,
            epochs=2,
            hidden_dim=16,
            time_embedding_dim=8,
            batch_size=16,
            early_stopping_patience=2,
            seed=8,
        )
        samples = sample_innovations(model, 5, 3, seed=9)
        self.assertEqual(samples.shape, (5, 3, 3))
        self.assertTrue(np.isfinite(samples).all())
        self.assertIn("best_epoch", model["history"])
        self.assertGreater(len(model["calibration_multipliers"]), 0)

    def test_calibration_multiplier_expands_underdispersed_draws(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(scale=0.2, size=(100, 50, 1))
        truth = rng.normal(scale=1.0, size=(100, 1))
        self.assertGreater(fit_scale_multiplier(samples, truth), 1.0)

    def test_forecast_calibration_scales_only_requested_lead(self):
        paths = np.array([
            [[-1.0], [-2.0]],
            [[0.0], [0.0]],
            [[1.0], [2.0]],
        ])
        adjusted = apply_forecast_calibration(paths, np.array([1.0, 2.0]))
        np.testing.assert_allclose(adjusted[:, 0], paths[:, 0])
        np.testing.assert_allclose(adjusted[:, 1, 0], [-4.0, 0.0, 4.0])

    def test_regularization_shrinks_and_caps_multiplier(self):
        adjusted = regularize_anchor_multipliers(
            {1: 0.4, 20: 2.0}, shrinkage=0.5, bounds=(0.8, 1.25)
        )
        self.assertEqual(adjusted, {1: 0.8, 20: 1.25})

    def test_adaptive_calibration_uses_only_available_outcomes(self):
        rng = np.random.default_rng(12)
        record = {
            "draws": {1: rng.normal(scale=0.1, size=(20, 1))},
            "truths": {1: np.array([0.20])},
            "available_at": {1: 11},
        }
        fallback = np.ones(1)
        before = fit_adaptive_multipliers(
            [record], (1,), fallback, current_origin=10
        )
        after = fit_adaptive_multipliers(
            [record], (1,), fallback, current_origin=11
        )
        np.testing.assert_allclose(before, [1.0])
        self.assertGreater(after[0], 1.0)

    def test_high_level_model_smoke(self):
        rng = np.random.default_rng(5)
        y = rng.normal(scale=0.01, size=(100, 2))
        config = CDIVARConfig(
            context_lags=2,
            volatility_span=10,
            calibration_horizons=(1, 2),
            calibration_paths=3,
            diffusion_timesteps=2,
            diffusion_epochs=1,
            diffusion_hidden_dim=12,
            diffusion_time_embedding_dim=8,
            diffusion_batch_size=16,
            early_stopping_patience=1,
        )
        fitted = CDIVAR(config).fit(y[:70], y[70:90])
        forecast = fitted.forecast(2, n_paths=4, y_history=y[:90], seed=2)
        self.assertEqual(forecast.forecast_paths.shape, (4, 2, 2))
        self.assertGreater(fitted.innovation_model_["calibration_origin_count"], 0)
        self.assertEqual(
            set(fitted.innovation_model_["calibration_anchor_multipliers"]),
            {1, 2},
        )

    def test_calibration_does_not_feed_back_into_volatility_state(self):
        model = ConditionalResidualDenoiser(
            input_dim=2, context_dim=6, hidden_dim=8, time_embedding_dim=4
        )
        schedule = make_ddpm_schedule(timesteps=2)
        kwargs = dict(
            model=model,
            schedule=schedule,
            n_paths=8,
            horizon=2,
            history=np.zeros((2, 2)),
            variance=np.ones(2),
            log_mean=np.zeros(2),
            log_std=np.ones(2),
            decay=0.9,
            device="cpu",
            seed=17,
        )
        nominal = _sample_recursive(**kwargs, multipliers=np.ones(2))
        widened_first_lead = _sample_recursive(
            **kwargs, multipliers=np.array([2.0, 1.0])
        )
        self.assertFalse(np.allclose(nominal[:, 0], widened_first_lead[:, 0]))
        np.testing.assert_allclose(nominal[:, 1], widened_first_lead[:, 1])


if __name__ == "__main__":
    unittest.main()
