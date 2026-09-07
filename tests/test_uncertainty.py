import numpy as np

from innovcal.evaluation import paired_block_bootstrap, paired_seed_block_bootstrap
from innovcal.evaluation.uncertainty import moving_block_indices


def test_moving_block_indices_have_valid_length_and_range():
    indices = moving_block_indices(37, 5, np.random.default_rng(2))
    assert indices.shape == (37,)
    assert indices.min() >= 0 and indices.max() < 37


def test_identical_forecasts_have_zero_paired_intervals():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(40, 2))
    samples = rng.normal(size=(60, 40, 2))
    result = paired_block_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=20,
        seed=4,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["ci_lower"], 0.0)
    assert np.allclose(result["ci_upper"], 0.0)
    assert "coverage_error" in set(result["metric"])


def test_identical_multiseed_forecasts_have_zero_hierarchical_intervals():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(35, 2))
    samples = rng.normal(size=(3, 40, 35, 2))
    result = paired_seed_block_bootstrap(
        target,
        {"candidate": samples, "comparator": samples.copy()},
        [("candidate", "comparator")],
        np.eye(2),
        block_length=5,
        n_bootstrap=20,
        seed=6,
    )
    assert np.allclose(result["difference"], 0.0)
    assert np.allclose(result["ci_lower"], 0.0)
    assert np.allclose(result["ci_upper"], 0.0)
