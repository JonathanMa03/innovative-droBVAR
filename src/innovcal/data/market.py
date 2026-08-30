"""Market-data acquisition and cleaning for the financial case study."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def download_adjusted_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo Finance via yfinance.

    ``end`` follows yfinance's exclusive-end convention. The function requests
    unadjusted OHLC data explicitly and selects ``Adj Close`` so the adjustment
    choice remains visible and stable across yfinance default changes.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Market-data support requires `pip install -e '.[market]'`."
        ) from exc

    names = tuple(str(ticker).upper() for ticker in tickers)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("tickers must contain at least two unique symbols.")

    downloaded = yf.download(
        list(names),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        repair=True,
        keepna=True,
        progress=False,
        threads=True,
        group_by="column",
        multi_level_index=True,
    )
    if downloaded.empty:
        raise ValueError("No market data were returned for the requested window.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Adj Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("Downloaded data do not contain adjusted close prices.")
        prices = downloaded["Adj Close"].copy()
    else:
        if len(names) != 1 or "Adj Close" not in downloaded.columns:
            raise ValueError("Unexpected yfinance column structure.")
        prices = downloaded[["Adj Close"]].rename(columns={"Adj Close": names[0]})

    return prices.reindex(columns=list(names))


def clean_adjusted_prices(
    prices: pd.DataFrame,
    maximum_missing_fraction: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate, align, and convert adjusted prices into log returns.

    Returns a complete-case adjusted-price panel and its one-period log returns.
    Missingness is reported by the caller before complete-case alignment; a
    ticker exceeding the configured threshold causes a failure rather than
    silent imputation of market prices.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame.")
    if prices.index.has_duplicates:
        raise ValueError("prices contain duplicate dates.")

    numeric = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    numeric = numeric.loc[~numeric.index.duplicated(keep="first")]
    missing_fraction = numeric.isna().mean()
    excessive = missing_fraction[missing_fraction > maximum_missing_fraction]
    if not excessive.empty:
        detail = ", ".join(f"{name}={value:.2%}" for name, value in excessive.items())
        raise ValueError(f"Excessive missing adjusted prices: {detail}")

    aligned = numeric.dropna(how="any")
    if aligned.empty or (aligned <= 0).any().any():
        raise ValueError("Aligned adjusted prices must be non-empty and positive.")

    returns = np.log(aligned).diff().dropna(how="any")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Cleaned returns contain non-finite values.")
    return aligned, returns
