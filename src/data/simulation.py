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

def simulate_var_with_innovations(
    A: np.ndarray,
    innovations: np.ndarray,
    burn_in: int = 0,
) -> np.ndarray:
    total, k = innovations.shape
    y = np.zeros((total, k))

    for t in range(1, total):
        y[t] = A @ y[t - 1] + innovations[t]

    if burn_in > 0:
        return y[burn_in:]

    return y


def generate_gaussian_innovations(
    n_obs: int,
    Sigma: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = Sigma.shape[0]

    return rng.multivariate_normal(
        mean=np.zeros(k),
        cov=Sigma,
        size=n_obs,
    )


def generate_student_t_innovations(
    n_obs: int,
    Sigma: np.ndarray,
    df: float = 5,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = Sigma.shape[0]

    z = rng.multivariate_normal(
        mean=np.zeros(k),
        cov=Sigma,
        size=n_obs,
    )

    g = rng.chisquare(df=df, size=(n_obs, 1))

    return z / np.sqrt(g / df)


def generate_mixture_innovations(
    n_obs: int,
    Sigma_low: np.ndarray,
    Sigma_high: np.ndarray,
    high_prob: float = 0.10,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = Sigma_low.shape[0]

    states = rng.binomial(1, high_prob, size=n_obs)
    innovations = np.zeros((n_obs, k))

    for t in range(n_obs):
        Sigma = Sigma_high if states[t] == 1 else Sigma_low
        innovations[t] = rng.multivariate_normal(
            mean=np.zeros(k),
            cov=Sigma,
        )

    return innovations


def generate_heteroskedastic_innovations(
    n_obs: int,
    k: int,
    base_scale: float = 0.5,
    high_scale: float = 1.8,
    period: int = 50,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    innovations = np.zeros((n_obs, k))

    for t in range(n_obs):
        scale = high_scale if (t // period) % 2 == 1 else base_scale
        innovations[t] = rng.normal(0, scale, size=k)

    return innovations