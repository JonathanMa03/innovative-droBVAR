def extract_forecast_residuals(
    datasets,
    lags,
    rnn_context_length,
    rnn_prediction_length,
    rnn_hidden_dim,
    rnn_num_layers,
    rnn_dropout,
    rnn_batch_size,
    rnn_epochs,
    rnn_lr,
    rnn_device,
):
    from src.api.modeling import fit_model, extract_residuals

    var_residual_store = {}
    rnn_residual_store = {}

    for name, data in datasets.items():
        y_train = data["y_train"]

        var_fit = fit_model(
            data=y_train,
            model="var",
            lags=lags,
            include_intercept=True,
        )

        var_residuals = extract_residuals(
            var_fit,
            model="var",
        )

        rnn_fit = fit_model(
            data=y_train,
            model="rnn",
            context_length=rnn_context_length,
            prediction_length=rnn_prediction_length,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            batch_size=rnn_batch_size,
            epochs=rnn_epochs,
            lr=rnn_lr,
            device=rnn_device,
            verbose=False,
        )

        rnn_residuals = extract_residuals(
            rnn_fit,
            model="rnn",
        )

        var_residual_store[name] = var_residuals
        rnn_residual_store[name] = rnn_residuals

        print(
            name,
            "VAR:",
            var_residuals.shape,
            "RNN:",
            rnn_residuals.shape,
        )

    return {
        "VAR": var_residual_store,
        "RNN": rnn_residual_store,
    }

import pandas as pd

from src.innovations.diagnostics import summarize_innovations


def residual_diagnostics_table(
    var_residual_store,
    rnn_residual_store,
):
    rows = []

    for forecast_model, store in [
        ("VAR", var_residual_store),
        ("RNN", rnn_residual_store),
    ]:

        for name, residuals in store.items():

            diag = summarize_innovations(
                residuals,
            )

            k = residuals.shape[1]

            for j in range(k):

                rows.append(
                    {
                        "dgp": name,
                        "forecast_model": forecast_model,
                        "series": j + 1,
                        "n_residuals": len(residuals),
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

def residual_diagnostics_summary(
    residual_diagnostics_df,
):
    return (
        residual_diagnostics_df
        .groupby(
            ["dgp", "forecast_model"]
        )[
            [
                "mean",
                "std",
                "skewness",
                "kurtosis",
                "excess_kurtosis",
                "jb_pvalue",
            ]
        ]
        .mean()
        .reset_index()
    )

def residual_acf_summary(
    var_residual_store,
    rnn_residual_store,
):
    import numpy as np
    import pandas as pd

    rows = []

    for forecast_model, store in [
        ("VAR", var_residual_store),
        ("RNN", rnn_residual_store),
    ]:
        for name, residuals in store.items():

            x = residuals[:, 0]

            acf1 = np.corrcoef(
                x[:-1],
                x[1:],
            )[0, 1]

            acf5 = np.corrcoef(
                x[:-5],
                x[5:],
            )[0, 1]

            rows.append(
                {
                    "dgp": name,
                    "forecast_model": forecast_model,
                    "acf_lag1": acf1,
                    "acf_lag5": acf5,
                }
            )

    return pd.DataFrame(rows)

def rolling_variance_summary(
    var_residual_store,
    rnn_residual_store,
    window=30,
):
    import pandas as pd

    rows = []

    for forecast_model, store in [
        ("VAR", var_residual_store),
        ("RNN", rnn_residual_store),
    ]:
        for name, residuals in store.items():

            x = residuals[:, 0]

            rolling_std = (
                pd.Series(x)
                .rolling(window)
                .std()
                .dropna()
            )

            rows.append(
                {
                    "dgp": name,
                    "forecast_model": forecast_model,
                    "mean_rolling_std": rolling_std.mean(),
                    "std_rolling_std": rolling_std.std(),
                    "cv_rolling_std":
                        rolling_std.std()
                        / rolling_std.mean(),
                }
            )

    return pd.DataFrame(rows)