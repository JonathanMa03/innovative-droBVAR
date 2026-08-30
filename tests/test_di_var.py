import unittest

import numpy as np

from innovcal.di_var.model import DIVAR, DIVARConfig, DIVARForecast
from innovcal.di_var.residuals import rolling_var_residuals


class RollingResidualTests(unittest.TestCase):
    def test_output_shape_and_determinism(self):
        rng = np.random.default_rng(7)
        y = rng.normal(size=(40, 3))

        first = rolling_var_residuals(y, initial_window=20, lags=2)
        second = rolling_var_residuals(y, initial_window=20, lags=2)

        self.assertEqual(first.shape, (20, 3))
        np.testing.assert_allclose(first, second)

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            rolling_var_residuals(np.ones((10, 2)), initial_window=1, lags=1)


class ForecastResultTests(unittest.TestCase):
    def test_quantiles_have_horizon_by_series_shape(self):
        paths = np.arange(60, dtype=float).reshape(5, 4, 3)
        result = DIVARForecast(paths, np.zeros_like(paths), 4, 5)

        quantiles = result.quantiles((0.5,))

        self.assertEqual(quantiles[0.5].shape, (4, 3))


class DIVARSmokeTests(unittest.TestCase):
    def test_tiny_fit_and_forecast(self):
        rng = np.random.default_rng(11)
        y = rng.normal(scale=0.1, size=(36, 2))
        model = DIVAR(
            DIVARConfig(
                lags=1,
                residual_window=24,
                diffusion_timesteps=4,
                diffusion_epochs=1,
                diffusion_hidden_dim=16,
                diffusion_time_embedding_dim=8,
                diffusion_batch_size=8,
            )
        ).fit(y)

        forecast = model.forecast(horizon=3, n_paths=4, seed=5)

        self.assertEqual(forecast.forecast_paths.shape, (4, 3, 2))
        self.assertEqual(forecast.innovation_paths.shape, (4, 3, 2))


if __name__ == "__main__":
    unittest.main()
