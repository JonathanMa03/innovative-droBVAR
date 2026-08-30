import unittest

import numpy as np
import pandas as pd

from innovcal.data.financial import (
    chronological_split,
    make_demo_prices,
    prices_to_log_returns,
)


class FinancialDataTests(unittest.TestCase):
    def test_log_returns(self):
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 100.0]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        returns = prices_to_log_returns(prices)

        self.assertEqual(returns.shape, (2, 2))
        np.testing.assert_allclose(returns["A"], np.log(1.1))

    def test_chronological_split(self):
        values = np.arange(100, dtype=float).reshape(50, 2)

        split = chronological_split(values, 0.6, 0.2)

        self.assertEqual(len(split.train), 30)
        self.assertEqual(len(split.calibration), 10)
        self.assertEqual(len(split.test), 10)
        np.testing.assert_array_equal(split.test[0], values[40])

    def test_demo_prices_are_positive_and_reproducible(self):
        first = make_demo_prices(20, 4, seed=9)
        second = make_demo_prices(20, 4, seed=9)

        self.assertTrue((first > 0).all().all())
        pd.testing.assert_frame_equal(first, second)
