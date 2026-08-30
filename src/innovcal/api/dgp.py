import pandas as pd

from innovcal.innovations.diagnostics import summarize_innovations


def innovation_diagnostics_table(
    datasets,
):
    rows = []

    for name, data in datasets.items():

        innovations = data["innovations"]

        diag = summarize_innovations(
            innovations,
        )

        k = innovations.shape[1]

        for j in range(k):

            rows.append(
                {
                    "dgp": name,
                    "series": j + 1,
                    "mean": diag["mean"][j],
                    "std": diag["std"][j],
                    "skewness": diag["skewness"][j],
                    "kurtosis": diag["kurtosis"][j],
                    "excess_kurtosis": diag["excess_kurtosis"][j],
                    "jb_stat": diag["jarque_bera_stat"][j],
                    "jb_pvalue": diag["jarque_bera_pvalue"][j],
                }
            )

    return pd.DataFrame(rows)

def innovation_diagnostics_summary(
    diagnostics_df,
):
    return (
        diagnostics_df
        .groupby("dgp")[
            [
                "skewness",
                "kurtosis",
                "excess_kurtosis",
                "jb_pvalue",
            ]
        ]
        .mean()
        .reset_index()
    )