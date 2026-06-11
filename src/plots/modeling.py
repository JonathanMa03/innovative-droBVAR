import numpy as np
import matplotlib.pyplot as plt


def plot_mean_forecasts(
    datasets,
    forecasts,
    dgp_names,
    horizon,
    forecast_label,
    variable=0,
):
    fig, axes = plt.subplots(
        len(dgp_names),
        1,
        figsize=(11, 9),
        sharex=False,
    )

    for ax, name in zip(
        axes,
        dgp_names,
    ):
        y_train = datasets[name]["y_train"]
        y_test = datasets[name]["y_test"]

        x_train = np.arange(len(y_train))

        x_test = np.arange(
            len(y_train),
            len(y_train) + horizon,
        )

        ax.plot(
            x_train,
            y_train[:, variable],
            label="train",
        )

        ax.plot(
            np.arange(
                len(y_train),
                len(y_train) + len(y_test),
            ),
            y_test[:, variable],
            label="test",
            alpha=0.5,
        )

        ax.plot(
            x_test,
            forecasts[name][:, variable],
            linestyle="--",
            label=forecast_label,
        )

        ax.set_title(
            f"{name}: {forecast_label}, Series {variable + 1}"
        )

        ax.legend()

    plt.tight_layout()

    return fig

def plot_rnn_training_loss(
    rnn_fits,
    dgp_names,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 6),
    )

    axes = axes.flatten()

    for ax, name in zip(
        axes,
        dgp_names,
    ):
        history = rnn_fits[name]["history"]

        ax.plot(
            history["loss"]
        )

        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")

    plt.tight_layout()

    return fig