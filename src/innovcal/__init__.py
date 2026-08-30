"""Calibrated and robust multivariate forecasting tools."""

__all__ = ["DIVAR", "DIVARConfig", "DIVARForecast"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Load optional neural components only when explicitly requested."""
    if name in __all__:
        from innovcal.di_var import DIVAR, DIVARConfig, DIVARForecast

        return {
            "DIVAR": DIVAR,
            "DIVARConfig": DIVARConfig,
            "DIVARForecast": DIVARForecast,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
