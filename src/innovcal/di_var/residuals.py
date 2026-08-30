"""Leakage-aware residual construction for innovation-model fitting."""

import numpy as np

from innovcal.vector_ar.fit import fit_var_ols
from innovcal.vector_ar.forecast import forecast_var_mean


def rolling_var_residuals(
    y: np.ndarray,
    initial_window: int,
    lags: int = 1,
    expanding: bool = True,
    refit_every: int = 1,
    include_intercept: bool = True,
) -> np.ndarray:
    """Return rolling one-step-ahead VAR forecast errors.

    At forecast origin ``t``, only observations strictly before ``t`` are used.
    With an expanding window, all available history is retained. With a rolling
    window, the fitting sample keeps length ``initial_window``.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 2:
        raise ValueError("y must have shape (n_observations, n_series).")
    if lags < 1:
        raise ValueError("lags must be at least 1.")
    if initial_window <= lags:
        raise ValueError("initial_window must be greater than lags.")
    if initial_window >= len(y):
        raise ValueError("initial_window must leave at least one forecast error.")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1.")

    residuals = []
    fitted = None

    for step, t in enumerate(range(initial_window, len(y))):
        if fitted is None or step % refit_every == 0:
            start = 0 if expanding else t - initial_window
            fitted = fit_var_ols(
                y[start:t],
                lags=lags,
                include_intercept=include_intercept,
            )

        prediction = forecast_var_mean(
            y_history=y[:t],
            beta=fitted["beta"],
            horizon=1,
            lags=lags,
            include_intercept=include_intercept,
        )[0]
        residuals.append(y[t] - prediction)

    return np.asarray(residuals)
