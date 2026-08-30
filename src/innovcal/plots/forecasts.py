import numpy as np
import matplotlib.pyplot as plt

from innovcal.forecasting.intervals import (
    prediction_interval,
    median_forecast,
)


def plot_forecast_grid_for_dgp(
    datasets,
    forecast_store,
    forecast_models,
    innovation_model_names,
    dgp_name,
    horizon,
    variable=0,
):
    fig, axes = plt.subplots(
        2,
        len(innovation_model_names),
        figsize=(18, 7),
        sharey=True,
    )

    for row, forecast_model in enumerate(forecast_models):
        for col, innovation_model in enumerate(innovation_model_names):

            ax = axes[row, col]

            y_train = datasets[dgp_name]["y_train"]
            y_true = datasets[dgp_name]["y_test"]

            paths = forecast_store[forecast_model][dgp_name][innovation_model][
                "forecast_paths"
            ]

            lower, upper = prediction_interval(
                paths,
                lower_q=0.05,
                upper_q=0.95,
            )

            median = median_forecast(paths)

            train_tail = y_train[-80:, variable]
            test_values = y_true[:, variable]

            x_train = np.arange(len(train_tail))
            x_forecast = np.arange(
                len(train_tail),
                len(train_tail) + horizon,
            )

            ax.plot(
                x_train,
                train_tail,
                label="train tail",
            )

            ax.plot(
                x_forecast,
                test_values,
                label="true future",
            )

            ax.plot(
                x_forecast,
                median[:, variable],
                linestyle="--",
                label="forecast median",
            )

            ax.fill_between(
                x_forecast,
                lower[:, variable],
                upper[:, variable],
                alpha=0.25,
            )

            if row == 0:
                ax.set_title(innovation_model)

            if col == 0:
                ax.set_ylabel(forecast_model)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    return fig