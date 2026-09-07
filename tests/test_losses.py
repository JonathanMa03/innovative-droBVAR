import torch

from innovcal.ca_rnn.losses import (
    calibration_loss,
    gaussian_nll,
    make_projection_matrix,
    projected_pits,
    sequential_calibration_loss,
)


def test_projected_calibration_objective_backpropagates():
    torch.manual_seed(4)
    target = torch.randn(64, 3)
    mean = torch.zeros(64, 3, requires_grad=True)
    scale = torch.eye(3).repeat(64, 1, 1).requires_grad_()
    projections = make_projection_matrix(3, n_random=2)
    pits = projected_pits(target, mean, scale, projections)
    loss = calibration_loss(pits) + sequential_calibration_loss(pits, max_lag=3)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_joint_nll_penalizes_bad_location():
    target = torch.zeros(20, 2)
    scale = torch.eye(2).repeat(20, 1, 1)
    assert gaussian_nll(target, torch.zeros_like(target), scale) < gaussian_nll(
        target, torch.full_like(target, 3.0), scale
    )


def test_projection_matrix_is_reproducible_and_normalized():
    first = make_projection_matrix(4, n_random=5, seed=10)
    second = make_projection_matrix(4, n_random=5, seed=10)
    assert torch.allclose(first, second)
    assert torch.allclose(first.norm(dim=1), torch.ones(len(first)))
