import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_residual_histograms(
    var_residual_store,
    rnn_residual_store,
    dgp_names,
    variable=0,
):
    fig, axes = plt.subplots(
        2,
        len(dgp_names),
        figsize=(16, 8),
        sharey=True,
    )

    for col, name in enumerate(dgp_names):

        # VAR
        residuals = var_residual_store[name]
        x = residuals[:, variable]

        ax = axes[0, col]

        ax.hist(
            x,
            bins=35,
            density=True,
            alpha=0.65,
        )

        grid = np.linspace(
            x.min(),
            x.max(),
            300,
        )

        ax.plot(
            grid,
            stats.norm.pdf(
                grid,
                loc=x.mean(),
                scale=x.std(ddof=1),
            ),
            linestyle="--",
        )

        ax.set_title(
            f"{name}\nVAR"
        )

        # RNN
        residuals = rnn_residual_store[name]
        x = residuals[:, variable]

        ax = axes[1, col]

        ax.hist(
            x,
            bins=35,
            density=True,
            alpha=0.65,
        )

        grid = np.linspace(
            x.min(),
            x.max(),
            300,
        )

        ax.plot(
            grid,
            stats.norm.pdf(
                grid,
                loc=x.mean(),
                scale=x.std(ddof=1),
            ),
            linestyle="--",
        )

        ax.set_title(
            f"{name}\nRNN"
        )

    plt.tight_layout()

    return fig

def plot_residual_qqplots(
    var_residual_store,
    rnn_residual_store,
    dgp_names,
    variable=0,
):
    fig, axes = plt.subplots(
        2,
        len(dgp_names),
        figsize=(16, 8),
    )

    for col, name in enumerate(dgp_names):

        # VAR
        ax = axes[0, col]

        stats.probplot(
            var_residual_store[name][:, variable],
            dist="norm",
            plot=ax,
        )

        ax.set_title(
            f"{name}\nVAR"
        )

        # RNN
        ax = axes[1, col]

        stats.probplot(
            rnn_residual_store[name][:, variable],
            dist="norm",
            plot=ax,
        )

        ax.set_title(
            f"{name}\nRNN"
        )

    plt.tight_layout()

    return fig

def plot_residual_acf(
    var_residual_store,
    rnn_residual_store,
    dgp_names,
    variable=0,
    max_lag=20,
):
    fig, axes = plt.subplots(
        2,
        len(dgp_names),
        figsize=(16, 8),
        sharey=True,
    )

    for col, name in enumerate(dgp_names):

        # VAR
        x = var_residual_store[name][:, variable]

        acfs = []

        for lag in range(1, max_lag + 1):
            acfs.append(
                np.corrcoef(
                    x[:-lag],
                    x[lag:],
                )[0, 1]
            )

        ax = axes[0, col]

        ax.bar(
            np.arange(1, max_lag + 1),
            acfs,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_title(
            f"{name}\nVAR"
        )

        # RNN
        x = rnn_residual_store[name][:, variable]

        acfs = []

        for lag in range(1, max_lag + 1):
            acfs.append(
                np.corrcoef(
                    x[:-lag],
                    x[lag:],
                )[0, 1]
            )

        ax = axes[1, col]

        ax.bar(
            np.arange(1, max_lag + 1),
            acfs,
        )

        ax.axhline(
            0,
            linewidth=0.8,
        )

        ax.set_title(
            f"{name}\nRNN"
        )

    plt.tight_layout()

    return fig

def plot_rolling_residual_std(
    var_residual_store,
    rnn_residual_store,
    dgp_names,
    variable=0,
    window=30,
):
    fig, axes = plt.subplots(
        2,
        len(dgp_names),
        figsize=(16, 8),
        sharey=True,
    )

    for col, name in enumerate(dgp_names):

        # VAR
        x = var_residual_store[name][:, variable]

        rolling_std = (
            pd.Series(x)
            .rolling(window)
            .std()
        )

        ax = axes[0, col]

        ax.plot(rolling_std)

        ax.set_title(
            f"{name}\nVAR"
        )

        # RNN
        x = rnn_residual_store[name][:, variable]

        rolling_std = (
            pd.Series(x)
            .rolling(window)
            .std()
        )

        ax = axes[1, col]

        ax.plot(rolling_std)

        ax.set_title(
            f"{name}\nRNN"
        )

    plt.tight_layout()

    return fig