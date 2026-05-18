import numpy as np


def make_stable_var_matrix(
    k: int,
    scale: float = 0.45,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, size=(k, k))

    eigvals = np.linalg.eigvals(A)
    max_abs = np.max(np.abs(eigvals))

    if max_abs == 0:
        raise ValueError("Generated zero matrix; retry with a different seed.")

    return A / max_abs * scale


def simulate_var_with_innovations(
    A: np.ndarray,
    innovations: np.ndarray,
    burn_in: int = 0,
    y0: np.ndarray | None = None,
) -> np.ndarray:
    total, k = innovations.shape
    y = np.zeros((total, k))

    if y0 is not None:
        y[0] = y0

    for t in range(1, total):
        y[t] = A @ y[t - 1] + innovations[t]

    return y[burn_in:] if burn_in > 0 else y


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
    df: float = 3.5,
    seed: int | None = None,
) -> np.ndarray:
    if df <= 2:
        raise ValueError("df must be greater than 2 for finite covariance.")

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
    high_prob: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    if not 0 <= high_prob <= 1:
        raise ValueError("high_prob must be between 0 and 1.")

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
    base_scale: float = 0.4,
    high_scale: float = 2.5,
    period: int = 40,
    seed: int | None = None,
) -> np.ndarray:
    if period < 1:
        raise ValueError("period must be at least 1.")

    rng = np.random.default_rng(seed)
    innovations = np.zeros((n_obs, k))

    for t in range(n_obs):
        scale = high_scale if (t // period) % 2 == 1 else base_scale
        innovations[t] = rng.normal(0, scale, size=k)

    return innovations


def generate_var_dataset(
    dgp_name: str,
    n_obs: int,
    burn_in: int,
    A: np.ndarray,
    Sigma: np.ndarray,
    seed: int | None = None,
    student_t_df: float = 3.5,
    mixture_high_prob: float = 0.15,
    mixture_high_scale: float = 9.0,
    hetero_base_scale: float = 0.4,
    hetero_high_scale: float = 2.5,
    hetero_period: int = 40,
) -> dict:
    total = n_obs + burn_in
    k = Sigma.shape[0]

    if dgp_name == "gaussian":
        innovations = generate_gaussian_innovations(
            n_obs=total,
            Sigma=Sigma,
            seed=seed,
        )

    elif dgp_name == "student_t":
        innovations = generate_student_t_innovations(
            n_obs=total,
            Sigma=Sigma,
            df=student_t_df,
            seed=seed,
        )

    elif dgp_name == "mixture":
        innovations = generate_mixture_innovations(
            n_obs=total,
            Sigma_low=Sigma,
            Sigma_high=mixture_high_scale * Sigma,
            high_prob=mixture_high_prob,
            seed=seed,
        )

    elif dgp_name == "heteroskedastic":
        innovations = generate_heteroskedastic_innovations(
            n_obs=total,
            k=k,
            base_scale=hetero_base_scale,
            high_scale=hetero_high_scale,
            period=hetero_period,
            seed=seed,
        )

    else:
        raise ValueError(
            "dgp_name must be one of: 'gaussian', 'student_t', 'mixture', 'heteroskedastic'."
        )

    y = simulate_var_with_innovations(
        A=A,
        innovations=innovations,
        burn_in=burn_in,
    )

    used_innovations = innovations[burn_in:]

    return {
        "name": dgp_name,
        "y": y,
        "innovations": used_innovations,
        "A": A,
        "Sigma": Sigma,
        "params": {
            "student_t_df": student_t_df,
            "mixture_high_prob": mixture_high_prob,
            "mixture_high_scale": mixture_high_scale,
            "hetero_base_scale": hetero_base_scale,
            "hetero_high_scale": hetero_high_scale,
            "hetero_period": hetero_period,
        },
    }


def generate_multiple_var_datasets(
    dgp_names: list[str],
    n_obs: int,
    burn_in: int,
    A: np.ndarray,
    Sigma: np.ndarray,
    base_seed: int = 123,
    **kwargs,
) -> dict:
    datasets = {}

    for i, name in enumerate(dgp_names):
        datasets[name] = generate_var_dataset(
            dgp_name=name,
            n_obs=n_obs,
            burn_in=burn_in,
            A=A,
            Sigma=Sigma,
            seed=base_seed + i,
            **kwargs,
        )

    return datasets