import torch

from innovcal.ca_rnn import CARNN, BayesianCARNN, CARNNConfig, predictive_nll


def test_deterministic_model_returns_valid_joint_distribution():
    model = CARNN(CARNNConfig(input_dim=4, hidden_dim=8))
    mean, scale = model(torch.randn(7, 12, 4))
    assert mean.shape == (7, 4)
    assert scale.shape == (7, 4, 4)
    assert torch.all(torch.diagonal(scale, dim1=-2, dim2=-1) > 0)
    assert torch.allclose(scale, torch.tril(scale))


def test_bayesian_model_has_positive_kl_and_stochastic_weights():
    model = BayesianCARNN(CARNNConfig(input_dim=3, hidden_dim=6))
    context = torch.randn(5, 10, 3)
    first, _ = model(context, sample=True)
    second, _ = model(context, sample=True)
    posterior_mean, _ = model(context, sample=False)
    repeated_mean, _ = model(context, sample=False)
    assert model.kl_divergence().item() > 0
    assert not torch.allclose(first, second)
    assert torch.allclose(posterior_mean, repeated_mean)


def test_predictive_nll_is_finite_for_both_model_types():
    context = torch.randn(6, 8, 2)
    target = torch.randn(6, 2)
    config = CARNNConfig(input_dim=2, hidden_dim=4)
    assert torch.isfinite(torch.tensor(predictive_nll(CARNN(config), context, target)))
    assert torch.isfinite(
        torch.tensor(predictive_nll(BayesianCARNN(config), context, target, 3))
    )
