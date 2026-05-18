import numpy as np


def simulate_var(
    A: np.ndarray,
    Sigma: np.ndarray,
    n_obs: int,
    burn_in: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a VAR(1):

        y_t = A y_{t-1} + u_t,
        u_t ~ N(0, Sigma)

    Returns
    -------
    y : np.ndarray
        Array of shape (n_obs, k).
    """
    rng = np.random.default_rng(seed)

    k = A.shape[0]
    total = n_obs + burn_in

    y = np.zeros((total, k))
    shocks = rng.multivariate_normal(
        mean=np.zeros(k),
        cov=Sigma,
        size=total,
    )

    for t in range(1, total):
        y[t] = A @ y[t - 1] + shocks[t]

    return y[burn_in:]


def make_stable_var_matrix(k: int, scale: float = 0.4, seed: int | None = None) -> np.ndarray:
    """
    Create a random stable VAR(1) coefficient matrix.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(k, k))

    eigvals = np.linalg.eigvals(A)
    max_abs = np.max(np.abs(eigvals))

    A = A / max_abs * scale

    return A