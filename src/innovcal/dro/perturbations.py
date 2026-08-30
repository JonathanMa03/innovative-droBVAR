import numpy as np


def variance_inflation_perturbation(
    samples: np.ndarray,
    epsilon: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inflate residual/innovation variance around the empirical center.

    epsilon = 0.10 means deviations from the center are multiplied by 1.10.
    """
    samples = np.asarray(samples, dtype=float)

    if center is None:
        center = samples.mean(axis=0)

    scale_factor = 1.0 + epsilon

    return center + scale_factor * (samples - center)


def tail_inflation_perturbation(
    samples: np.ndarray,
    epsilon: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Push samples radially away from the empirical center.

    This stresses tail behavior while approximately preserving directions.
    """
    samples = np.asarray(samples, dtype=float)

    if center is None:
        center = samples.mean(axis=0)

    direction = samples - center

    norms = np.linalg.norm(
        direction,
        axis=1,
        keepdims=True,
    )

    safe_norms = np.where(
        norms == 0.0,
        1.0,
        norms,
    )

    unit_direction = direction / safe_norms

    return samples + epsilon * unit_direction


def outlier_contamination_perturbation(
    samples: np.ndarray,
    epsilon: float,
    contamination_scale: float = 4.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Replace an epsilon fraction of samples with extreme contaminated draws.

    epsilon is interpreted as contamination probability.
    For example, epsilon = 0.10 replaces about 10% of innovations.

    Contaminated draws are generated from a widened Gaussian using the
    empirical mean and covariance of the original samples.
    """
    samples = np.asarray(samples, dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, k).")

    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be between 0 and 1 for contamination.")

    rng = np.random.default_rng(seed)

    n_samples, k = samples.shape

    contaminated = samples.copy()

    mask = rng.uniform(size=n_samples) < epsilon

    n_contaminated = int(mask.sum())

    if n_contaminated == 0:
        return contaminated

    mean = samples.mean(axis=0)

    cov = np.cov(
        samples,
        rowvar=False,
    )

    cov = np.atleast_2d(cov)
    cov = cov + 1e-6 * np.eye(k)

    outliers = rng.multivariate_normal(
        mean=mean,
        cov=(contamination_scale ** 2) * cov,
        size=n_contaminated,
    )

    contaminated[mask] = outliers

    return contaminated


def additive_gaussian_perturbation(
    samples: np.ndarray,
    epsilon: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Backward-compatible local Gaussian noise perturbation.

    Not used as the main thesis stress test, but kept for compatibility.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)

    noise = rng.normal(
        loc=0.0,
        scale=epsilon,
        size=samples.shape,
    )

    return samples + noise


def perturb_innovation_paths(
    innovation_paths: np.ndarray,
    method: str,
    epsilon: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Perturb innovation paths with shape (n_paths, horizon, k).

    Supported thesis-facing methods:
        variance_inflation
        tail_inflation
        outlier_contamination

    Backward-compatible aliases:
        scale -> variance_inflation
        radial_tail -> tail_inflation
        gaussian -> additive_gaussian
    """
    innovation_paths = np.asarray(
        innovation_paths,
        dtype=float,
    )

    if innovation_paths.ndim != 3:
        raise ValueError(
            "innovation_paths must have shape (n_paths, horizon, k)."
        )

    n_paths, horizon, k = innovation_paths.shape

    flat = innovation_paths.reshape(
        n_paths * horizon,
        k,
    )

    method = method.lower()

    if method in {
        "variance_inflation",
        "scale",
    }:
        perturbed = variance_inflation_perturbation(
            flat,
            epsilon=epsilon,
        )

    elif method in {
        "tail_inflation",
        "radial_tail",
    }:
        perturbed = tail_inflation_perturbation(
            flat,
            epsilon=epsilon,
        )

    elif method in {
        "outlier_contamination",
        "contamination",
    }:
        perturbed = outlier_contamination_perturbation(
            flat,
            epsilon=epsilon,
            seed=seed,
        )

    elif method in {
        "gaussian",
        "additive_gaussian",
    }:
        perturbed = additive_gaussian_perturbation(
            flat,
            epsilon=epsilon,
            seed=seed,
        )

    else:
        raise ValueError(
            "method must be one of: "
            "'variance_inflation', 'tail_inflation', "
            "'outlier_contamination'."
        )

    return perturbed.reshape(
        n_paths,
        horizon,
        k,
    )