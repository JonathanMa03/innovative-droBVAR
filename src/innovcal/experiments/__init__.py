"""Reproducible experiment workflows."""

__all__ = ["FinancialExperimentResult", "run_financial_experiment"]


def __getattr__(name: str):
    """Avoid importing experiment workflows during low-level artifact imports."""
    if name in __all__:
        from innovcal.experiments.financial import (
            FinancialExperimentResult,
            run_financial_experiment,
        )

        return {
            "FinancialExperimentResult": FinancialExperimentResult,
            "run_financial_experiment": run_financial_experiment,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
