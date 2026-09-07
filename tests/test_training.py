import numpy as np
import torch

from innovcal.ca_rnn import CARNN, CARNNConfig, TrainingConfig, fit_model
from innovcal.data import WindowDataset


def test_training_smoke_run_updates_and_restores_model():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(100, 2)).astype(np.float32)
    train = WindowDataset(values[:70], history=8)
    validation = WindowDataset(values[70:], history=8)
    model = CARNN(CARNNConfig(input_dim=2, hidden_dim=6))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    result = fit_model(
        model,
        train,
        validation,
        TrainingConfig(epochs=2, batch_size=16, patience=2, lambda_cal=1.0),
    )
    assert len(result.history) == 2
    assert 1 <= result.best_epoch <= 2
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_fixed_epoch_mode_runs_full_budget_and_selects_final_model():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    model = CARNN(CARNNConfig(input_dim=2, hidden_dim=4))
    result = fit_model(
        model,
        WindowDataset(values[:55], history=6),
        WindowDataset(values[55:], history=6),
        TrainingConfig(epochs=3, batch_size=16, patience=1, early_stopping=False),
    )
    assert len(result.history) == 3
    assert result.best_epoch == 3
    assert not result.stopped_early
