import numpy as np


def additive_gaussian_perturbation(
    samples: np.ndarray,
    epsilon: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Add isotropic Gaussian perturbation to samples.

    This is a simple practical proxy for local distributional perturbation.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)

    noise = rng.normal(
        loc=0.0,
        scale=epsilon,
        size=samples.shape,
    )

    return samples + noise


def radial_tail_perturbation(
    samples: np.ndarray,
    epsilon: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Push samples radially away from the center.

    Useful for tail stress testing.
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

    safe_norms = np.where(norms == 0, 1.0, norms)
    unit_direction = direction / safe_norms

    return samples + epsilon * unit_direction


def scale_perturbation(
    samples: np.ndarray,
    scale_factor: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inflate or shrink samples around their center.
    """
    samples = np.asarray(samples, dtype=float)

    if center is None:
        center = samples.mean(axis=0)

    return center + scale_factor * (samples - center)


def perturb_innovation_paths(
    innovation_paths: np.ndarray,
    method: str,
    epsilon: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Perturb innovation paths with shape (n_paths, horizon, k).
    """
    innovation_paths = np.asarray(innovation_paths, dtype=float)

    if innovation_paths.ndim != 3:
        raise ValueError("innovation_paths must have shape (n_paths, horizon, k).")

    n_paths, horizon, k = innovation_paths.shape

    flat = innovation_paths.reshape(n_paths * horizon, k)

    if method == "gaussian":
        perturbed = additive_gaussian_perturbation(
            flat,
            epsilon=epsilon,
            seed=seed,
        )

    elif method == "radial_tail":
        perturbed = radial_tail_perturbation(
            flat,
            epsilon=epsilon,
        )

    elif method == "scale":
        perturbed = scale_perturbation(
            flat,
            scale_factor=1.0 + epsilon,
        )

    else:
        raise ValueError("method must be one of: 'gaussian', 'radial_tail', 'scale'.")

    return perturbed.reshape(n_paths, horizon, k)