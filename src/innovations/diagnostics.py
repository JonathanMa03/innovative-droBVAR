import numpy as np
from scipy import stats


def innovation_moments(
    innovations: np.ndarray,
) -> dict:
    innovations = np.asarray(innovations, dtype=float)

    return {
        "mean": innovations.mean(axis=0),
        "std": innovations.std(axis=0, ddof=1),
        "skewness": stats.skew(
            innovations,
            axis=0,
            bias=False,
        ),
        "kurtosis": stats.kurtosis(
            innovations,
            axis=0,
            fisher=False,
            bias=False,
        ),
        "excess_kurtosis": stats.kurtosis(
            innovations,
            axis=0,
            fisher=True,
            bias=False,
        ),
    }


def jarque_bera_by_series(
    innovations: np.ndarray,
) -> dict:
    innovations = np.asarray(innovations, dtype=float)

    statistics = []
    pvalues = []

    for j in range(innovations.shape[1]):
        result = stats.jarque_bera(innovations[:, j])
        statistics.append(result.statistic)
        pvalues.append(result.pvalue)

    return {
        "statistic": np.asarray(statistics),
        "pvalue": np.asarray(pvalues),
    }


def innovation_correlation(
    innovations: np.ndarray,
) -> np.ndarray:
    innovations = np.asarray(innovations, dtype=float)
    return np.corrcoef(innovations.T)


def summarize_innovations(
    innovations: np.ndarray,
) -> dict:
    moments = innovation_moments(innovations)
    jb = jarque_bera_by_series(innovations)

    return {
        **moments,
        "jarque_bera_stat": jb["statistic"],
        "jarque_bera_pvalue": jb["pvalue"],
        "correlation": innovation_correlation(innovations),
    }


def flatten_sampled_innovations(
    sampled_innovations: np.ndarray,
) -> np.ndarray:
    """
    Convert sampled forecast innovations from shape
    (n_paths, horizon, k) to (n_paths * horizon, k).
    """
    sampled_innovations = np.asarray(sampled_innovations, dtype=float)

    if sampled_innovations.ndim != 3:
        raise ValueError("sampled_innovations must have shape (n_paths, horizon, k).")

    n_paths, horizon, k = sampled_innovations.shape

    return sampled_innovations.reshape(n_paths * horizon, k)