import numpy as np
from scipy import stats


def residual_summary(
    residuals: np.ndarray,
) -> dict:
    return {
        "mean": residuals.mean(axis=0),
        "std": residuals.std(axis=0, ddof=1),
        "covariance": np.cov(residuals.T),
        "correlation": np.corrcoef(residuals.T),
        "min": residuals.min(axis=0),
        "max": residuals.max(axis=0),
    }


def residual_moments(
    residuals: np.ndarray,
) -> dict:
    return {
        "skewness": stats.skew(
            residuals,
            axis=0,
            bias=False,
        ),
        "kurtosis": stats.kurtosis(
            residuals,
            axis=0,
            fisher=False,
            bias=False,
        ),
        "excess_kurtosis": stats.kurtosis(
            residuals,
            axis=0,
            fisher=True,
            bias=False,
        ),
    }


def jarque_bera_test(
    residuals: np.ndarray,
) -> dict:
    stats_list = []
    pvalues = []

    for j in range(residuals.shape[1]):
        result = stats.jarque_bera(
            residuals[:, j]
        )

        stats_list.append(result.statistic)
        pvalues.append(result.pvalue)

    return {
        "statistic": np.asarray(stats_list),
        "pvalue": np.asarray(pvalues),
    }


def autocorrelation(
    x: np.ndarray,
    lag: int = 1,
) -> float:
    x = np.asarray(x)

    return np.corrcoef(
        x[:-lag],
        x[lag:],
    )[0, 1]


def residual_autocorrelation(
    residuals: np.ndarray,
    lag: int = 1,
) -> np.ndarray:
    return np.asarray([
        autocorrelation(
            residuals[:, j],
            lag=lag,
        )
        for j in range(residuals.shape[1])
    ])


def summarize_var_diagnostics(
    residuals: np.ndarray,
) -> dict:
    summary = residual_summary(residuals)
    moments = residual_moments(residuals)
    jb = jarque_bera_test(residuals)

    return {
        **summary,
        **moments,
        "jarque_bera_stat": jb["statistic"],
        "jarque_bera_pvalue": jb["pvalue"],
        "autocorrelation_lag1": residual_autocorrelation(
            residuals,
            lag=1,
        ),
    }