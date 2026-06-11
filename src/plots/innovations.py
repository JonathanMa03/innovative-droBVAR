import matplotlib.pyplot as plt


def plot_innovation_comparison(
    residual_store,
    sampled_innovations,
    innovation_model_names,
    forecast_model,
    dgp_name,
    variable=0,
):
    residuals = residual_store[
        forecast_model
    ][
        dgp_name
    ][
        :, variable
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(9, 9),
        sharex=False,
    )

    axes[0].hist(
        residuals,
        bins=35,
        density=True,
        alpha=0.65,
    )

    axes[0].set_title(
        f"{forecast_model} | {dgp_name}: empirical residuals"
    )

    for ax, method in zip(
        axes[1:],
        innovation_model_names,
    ):
        samples = sampled_innovations[
            forecast_model
        ][
            dgp_name
        ][
            method
        ]

        flat_samples = samples.reshape(
            -1,
            samples.shape[-1],
        )

        ax.hist(
            flat_samples[:, variable],
            bins=35,
            density=True,
            alpha=0.65,
        )

        ax.set_title(
            f"{forecast_model} | {dgp_name}: {method} innovation samples"
        )

    plt.tight_layout()

    return fig

import scipy.stats as stats


def plot_innovation_qq_comparison(
    residual_store,
    sampled_innovations,
    innovation_model_names,
    forecast_model,
    dgp_name,
    variable=0,
):
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7, 12),
    )

    residuals = residual_store[forecast_model][dgp_name][:, variable]

    stats.probplot(
        residuals,
        dist="norm",
        plot=axes[0],
    )

    axes[0].set_title(
        f"{forecast_model} | {dgp_name}: empirical residuals QQ"
    )

    for ax, method in zip(
        axes[1:],
        innovation_model_names,
    ):
        samples = sampled_innovations[forecast_model][dgp_name][method]

        flat_samples = samples.reshape(
            -1,
            samples.shape[-1],
        )

        stats.probplot(
            flat_samples[:, variable],
            dist="norm",
            plot=ax,
        )

        ax.set_title(
            f"{forecast_model} | {dgp_name}: {method} samples QQ"
        )

    plt.tight_layout()

    return fig