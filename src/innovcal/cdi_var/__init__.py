"""Conditional, volatility-aware, calibrated DI-VAR."""

__all__ = ["CDIVAR", "CDIVARConfig", "CDIVARForecast"]


def __getattr__(name):
    if name in __all__:
        from innovcal.cdi_var.model import CDIVAR, CDIVARConfig, CDIVARForecast
        return {"CDIVAR": CDIVAR, "CDIVARConfig": CDIVARConfig,
                "CDIVARForecast": CDIVARForecast}[name]
    raise AttributeError(name)
