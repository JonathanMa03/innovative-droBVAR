"""Market-data acquisition and cleaning for the financial application."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def download_adjusted_prices(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily closing prices through the optional yfinance dependency."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install market support with `pip install -e '.[market]'`.") from exc
    names = tuple(str(ticker).upper() for ticker in tickers)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("tickers must contain at least two unique symbols")
    downloaded = yf.download(
        list(names), start=start, end=end, auto_adjust=False, actions=False,
        repair=True, keepna=True, progress=False, threads=True,
    )
    if downloaded.empty:
        raise ValueError("no market data were returned")
    prices = downloaded["Adj Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
    return prices.reindex(columns=list(names))


def clean_adjusted_prices(
    prices: pd.DataFrame, maximum_missing_fraction: float = 0.01
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align positive prices and return the clean panel and log returns."""
    if not isinstance(prices, pd.DataFrame) or prices.empty or prices.index.has_duplicates:
        raise ValueError("prices must be a non-empty DataFrame with unique dates")
    numeric = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    excessive = numeric.isna().mean()
    excessive = excessive[excessive > maximum_missing_fraction]
    if not excessive.empty:
        raise ValueError(f"excessive missingness: {excessive.to_dict()}")
    aligned = numeric.dropna(how="any")
    if aligned.empty or (aligned <= 0).any().any():
        raise ValueError("aligned prices must be non-empty and positive")
    returns = np.log(aligned).diff().dropna(how="any")
    return aligned, returns
