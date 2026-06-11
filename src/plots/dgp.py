import matplotlib.pyplot as plt


def plot_raw_dgp_series(
    datasets,
    variable=0,
):
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(11, 8),
        sharex=True,
    )

    for ax, (name, data) in zip(
        axes,
        datasets.items(),
    ):
        y = data["y"]

        ax.plot(y[:, variable])

        ax.set_title(
            f"{name}: Series {variable + 1}"
        )

    plt.tight_layout()

    return fig

def plot_all_series(
    y,
    title,
):
    plt.figure(
        figsize=(11, 4),
    )

    k = y.shape[1]

    for j in range(k):
        plt.plot(
            y[:, j],
            label=f"Series {j + 1}",
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()

def plot_innovation_histograms(
    datasets,
    variable=0,
    bins=35,
):
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(9, 9),
        sharex=False,
    )

    for ax, (name, data) in zip(
        axes,
        datasets.items(),
    ):
        innovations = data["innovations"][:, variable]

        ax.hist(
            innovations,
            bins=bins,
            density=True,
            alpha=0.65,
        )

        ax.set_title(
            f"{name}: Innovation Distribution, Series {variable + 1}"
        )

    plt.tight_layout()

    return fig

import scipy.stats as stats


def plot_innovation_qqplots(
    datasets,
    variable=0,
):
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(7, 12),
    )

    for ax, (name, data) in zip(
        axes,
        datasets.items(),
    ):
        innovations = data["innovations"][:, variable]

        stats.probplot(
            innovations,
            dist="norm",
            plot=ax,
        )

        ax.set_title(
            f"QQ Plot: {name}, Series {variable + 1}"
        )

    plt.tight_layout()

    return fig