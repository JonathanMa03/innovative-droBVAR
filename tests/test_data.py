import numpy as np

from innovcal.data import (
    Standardizer,
    WindowDataset,
    chronological_split,
    partition_window_datasets,
)
from innovcal.data.simulation import simulate_multivariate_series


def test_chronological_pipeline_is_disjoint_and_train_scaled():
    values = simulate_multivariate_series(200, dimension=4, seed=7)
    split = chronological_split(values)
    assert len(split.train) + len(split.validation) + len(split.test) == 200
    scaler = Standardizer.fit(split.train)
    transformed = scaler.transform(split.train)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
    dataset = WindowDataset(transformed, history=12)
    context, target = dataset[0]
    assert context.shape == (12, 4)
    assert target.shape == (4,)


def test_shift_changes_only_post_shift_realizations_for_common_seed():
    base = simulate_multivariate_series(150, seed=19, shift_start=100, shift_severity=0)
    shifted = simulate_multivariate_series(150, seed=19, shift_start=100, shift_severity=0.5)
    assert np.allclose(base[:100], shifted[:100])
    assert not np.allclose(base[100:], shifted[100:])


def test_partition_windows_retain_all_evaluation_targets_without_leakage():
    values = np.arange(240, dtype=float).reshape(120, 2)
    split = chronological_split(values)
    scaler = Standardizer.fit(split.train)
    train, validation, test = partition_window_datasets(split, scaler, history=5)
    assert len(validation) == len(split.validation)
    assert len(test) == len(split.test)
    validation_context, validation_target = validation[0]
    test_context, test_target = test[0]
    assert np.allclose(validation_context[-1], scaler.transform(split.train)[-1])
    assert np.allclose(validation_target, scaler.transform(split.validation)[0])
    assert np.allclose(test_context[-1], scaler.transform(split.validation)[-1])
    assert np.allclose(test_target, scaler.transform(split.test)[0])
