import numpy as np

from innovcal.evaluation import (
    evaluate_samples,
    interval_components,
    pit_diagnostics,
    summarize_interval_components,
)
from innovcal.evaluation.metrics import empirical_pits


def test_sample_evaluation_returns_finite_metrics():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(30, 3))
    samples = rng.normal(size=(100, 30, 3))
    metrics = evaluate_samples(target, samples)
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["coverage"] <= 1


def test_uniform_grid_has_small_pit_discrepancy():
    grid = np.linspace(0.005, 0.995, 100)[:, None]
    diagnostics = pit_diagnostics(np.repeat(grid, 3, axis=1))
    assert diagnostics["pit_calibration_error"] < 1e-3


def test_projected_pits_include_dependence_directions():
    rng = np.random.default_rng(11)
    target = rng.normal(size=(20, 2))
    samples = rng.normal(size=(80, 20, 2))
    projections = np.array([[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]])
    pits = empirical_pits(target, samples, projections)
    assert pits.shape == (20, 3)
    metrics = evaluate_samples(target, samples, projections=projections)
    assert "pit_calibration_error" in metrics
    assert "coordinate_pit_calibration_error" in metrics


def test_interval_components_sum_to_interval_score():
    rng = np.random.default_rng(12)
    target = rng.normal(size=(25, 2))
    samples = rng.normal(size=(100, 25, 2))
    components = interval_components(target, samples)
    expected = (
        components["width"]
        + components["lower_penalty"]
        + components["upper_penalty"]
    )
    assert np.allclose(components["interval_score"], expected)
    summary = summarize_interval_components(target, samples, ["a", "b"])
    assert summary["asset"].tolist() == ["all", "a", "b"]
