import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_diffusion_training_losses(
    diffusion_models,
    forecast_models,
    dgp_names,
):
    fig, axes = plt.subplots(
        2,
        len(dgp_names),
        figsize=(16, 7),
        sharey=True,
    )

    for row, forecast_model in enumerate(
        forecast_models
    ):
        for col, dgp_name in enumerate(
            dgp_names
        ):
            losses = diffusion_models[
                forecast_model
            ][
                dgp_name
            ]["history"]["loss"]

            ax = axes[row, col]

            ax.plot(
                losses
            )

            ax.set_title(
                f"{forecast_model} | {dgp_name}"
            )

            ax.set_xlabel(
                "epoch"
            )

            if col == 0:
                ax.set_ylabel(
                    "loss"
                )

    plt.tight_layout()

    return fig

def plot_diffusion_histogram_grid(
    residual_store,
    diffusion_samples,
    dgp_names,
    variable=0,
):
    fig, axes = plt.subplots(
        4,
        len(dgp_names),
        figsize=(16, 14),
        sharex=False,
        sharey=False,
    )

    row_specs = [
        ("VAR", "Empirical"),
        ("VAR", "Diffusion"),
        ("RNN", "Empirical"),
        ("RNN", "Diffusion"),
    ]

    for row, (forecast_model, source) in enumerate(row_specs):
        for col, dgp_name in enumerate(dgp_names):

            if source == "Empirical":
                x = residual_store[forecast_model][dgp_name][:, variable]
            else:
                x = diffusion_samples[forecast_model][dgp_name][:, variable]

            axes[row, col].hist(
                x,
                bins=35,
                density=True,
                alpha=0.7,
            )

            if row == 0:
                axes[row, col].set_title(dgp_name)

            if col == 0:
                axes[row, col].set_ylabel(
                    f"{forecast_model}\n{source}"
                )

    plt.tight_layout()

    return fig

import scipy.stats as stats


def plot_diffusion_qq_grid(
    residual_store,
    diffusion_samples,
    dgp_names,
    variable=0,
):
    fig, axes = plt.subplots(
        4,
        len(dgp_names),
        figsize=(16, 14),
    )

    row_specs = [
        ("VAR", "Empirical"),
        ("VAR", "Diffusion"),
        ("RNN", "Empirical"),
        ("RNN", "Diffusion"),
    ]

    for row, (forecast_model, source) in enumerate(row_specs):
        for col, dgp_name in enumerate(dgp_names):

            if source == "Empirical":
                x = residual_store[
                    forecast_model
                ][
                    dgp_name
                ][
                    :, variable
                ]
            else:
                x = diffusion_samples[
                    forecast_model
                ][
                    dgp_name
                ][
                    :, variable
                ]

            stats.probplot(
                x,
                dist="norm",
                plot=axes[row, col],
            )

            if row == 0:
                axes[row, col].set_title(
                    dgp_name
                )

            if col == 0:
                axes[row, col].set_ylabel(
                    f"{forecast_model}\n{source}"
                )

    plt.tight_layout()

    return fig