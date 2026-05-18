import numpy as np
from scipy import stats


def innovation_moments(residuals: np.ndarray) -> dict:
    return {
        "mean": residuals.mean(axis=0),
        "std": residuals.std(axis=0, ddof=1),
        "skewness": stats.skew(residuals, axis=0, bias=False),
        "kurtosis": stats.kurtosis(residuals, axis=0, fisher=False, bias=False),
        "excess_kurtosis": stats.kurtosis(residuals, axis=0, fisher=True, bias=False),
    }


def jarque_bera_by_series(residuals: np.ndarray) -> dict:
    statistics = []
    pvalues = []

    for j in range(residuals.shape[1]):
        result = stats.jarque_bera(residuals[:, j])
        statistics.append(result.statistic)
        pvalues.append(result.pvalue)

    return {
        "statistic": np.array(statistics),
        "pvalue": np.array(pvalues),
    }


def residual_correlation(residuals: np.ndarray) -> np.ndarray:
    return np.corrcoef(residuals.T)


def summarize_innovation_diagnostics(residuals: np.ndarray) -> dict:
    moments = innovation_moments(residuals)
    jb = jarque_bera_by_series(residuals)

    return {
        **moments,
        "jarque_bera_stat": jb["statistic"],
        "jarque_bera_pvalue": jb["pvalue"],
        "correlation": residual_correlation(residuals),
    }