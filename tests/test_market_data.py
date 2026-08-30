import unittest

import numpy as np
import pandas as pd

from innovcal.data.market import clean_adjusted_prices


class MarketDataCleaningTests(unittest.TestCase):
    def test_aligns_prices_and_computes_returns(self):
        index = pd.date_range("2020-01-01", periods=4)
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 55.0, 60.5, 66.55]},
            index=index,
        )

        aligned, returns = clean_adjusted_prices(prices)

        self.assertEqual(aligned.shape, (4, 2))
        self.assertEqual(returns.shape, (3, 2))
        np.testing.assert_allclose(returns.to_numpy(), np.log(1.1))

    def test_rejects_excessive_missingness(self):
        prices = pd.DataFrame(
            {"A": [100.0, np.nan, 102.0], "B": [50.0, 51.0, 52.0]}
        )
        with self.assertRaises(ValueError):
            clean_adjusted_prices(prices, maximum_missing_fraction=0.1)


if __name__ == "__main__":
    unittest.main()
