import unittest

import numpy as np

from innovcal.experiments.financial import run_financial_experiment


class FinancialExperimentTests(unittest.TestCase):
    def test_classical_workflow(self):
        rng = np.random.default_rng(21)
        returns = rng.normal(scale=0.01, size=(80, 3))

        result = run_financial_experiment(
            returns,
            methods=("gaussian", "bootstrap"),
            n_paths=20,
            seed=4,
        )

        self.assertEqual(result.residuals.shape, (16, 3))
        self.assertEqual(set(result.forecasts), {"gaussian", "bootstrap"})
        self.assertEqual(result.forecasts["gaussian"]["forecast_paths"].shape, (20, 16, 3))
        self.assertEqual(len(result.evaluation), 2)


if __name__ == "__main__":
    unittest.main()
