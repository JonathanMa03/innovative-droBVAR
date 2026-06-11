from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    seed: int = 123

    dgp_names: list[str] = field(
        default_factory=lambda: [
            "gaussian",
            "student_t",
            "mixture",
            "heteroskedastic",
        ]
    )

    innovation_models: list[str] = field(
        default_factory=lambda: [
            "gaussian",
            "bootstrap",
            "student_t",
            "diffusion",
        ]
    )

    forecast_model: str = "VAR"

    lags: int = 1
    n_train: int = 300
    horizon: int = 40
    n_paths: int = 250

    interval: tuple[float, float] = (0.05, 0.95)
    nominal_levels: tuple[float, ...] = (0.5, 0.8, 0.9)

    diffusion_timesteps: int = 200
    diffusion_epochs: int = 1500
    diffusion_lr: float = 5e-4
    diffusion_hidden_dim: int = 256
    diffusion_time_embedding_dim: int = 64

    student_t_df: float = 5.0