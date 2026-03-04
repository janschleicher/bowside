from collections.abc import Callable
from typing import Literal

import pytest
import scanpy as sc
import torch

from sc_markov_ode.dataset import MarkovChainDataset
from sc_markov_ode.model import GRNModel, GRNModelSDE
from sc_markov_ode.training import integrate_model, qqloss, train_integration
from tests.conftest import TEST_DATA_PATH

torch.manual_seed(42)

Y_TRUE = torch.arange(1, 101).float().reshape((10, 10))
Y_PRED_PERFECT = Y_TRUE[:, torch.randperm(10)]
Y_PRED_TWO_OFF = (Y_TRUE + 2)[:, torch.randperm(10)]
Y_TRUE_SMALL = torch.tensor([[1, 1, 1, 1, 1, 3, 8, 8, 8, 8, 8]]).float()[
    :, torch.randperm(11)
]
Y_PRED_SMALL = torch.tensor([[3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6]]).float()[
    :, torch.randperm(11)
]


@pytest.mark.parametrize(
    "y_pred, y_true, metric, expected_loss",
    [
        [Y_PRED_PERFECT, Y_TRUE, torch.nn.functional.l1_loss, 0],
        [Y_PRED_PERFECT, Y_TRUE, torch.nn.functional.mse_loss, 0],
        [Y_PRED_TWO_OFF, Y_TRUE, torch.nn.functional.l1_loss, 2],
        [Y_PRED_TWO_OFF, Y_TRUE, torch.nn.functional.mse_loss, 4],
        [Y_PRED_SMALL, Y_TRUE_SMALL, torch.nn.functional.l1_loss, 2],
        [Y_PRED_SMALL, Y_TRUE_SMALL, torch.nn.functional.mse_loss, 4],
    ],
)
def test_qqloss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    expected_loss: float,
):
    loss = qqloss(y_pred, y_true, dim=1, n_quantiles=10, metric=metric)
    assert loss == expected_loss, (
        f"Wrong loss value. Expected: {expected_loss:.2f}, actual: {loss.item():.2f}"
    )
    assert loss != metric(y_pred, y_true), (
        "The qqloss should not be the same as the underlying metric for the given test cases."
    )


@pytest.mark.parametrize(
    "model_type, batch_size, num_chains, shared_sigma, fixed_sigma, constant_noise",
    [
        ["ode", None, 100, None, None, None],
        ["ode", 100, 500, None, None, None],
        ["sde", None, 100, True, True, True],
        ["sde", None, 100, True, True, False],
        ["sde", None, 100, True, False, True],
        ["sde", None, 100, False, False, True],
    ],
)
def test_train_integration(
    model_type: Literal["ode", "sde"],
    batch_size: int,
    num_chains: int,
    shared_sigma: bool,
    fixed_sigma: bool,
    constant_noise: bool,
):
    torch.manual_seed(42)
    dt = 0.1
    atol = 0.1
    rtol = 0.1
    solver_method = None
    if model_type == "ode" and batch_size:
        solver_method = "adaptive_heun"
    elif model_type == "sde":
        solver_method = "milstein"
    loss_metric = torch.nn.functional.mse_loss

    # Select GPU if available
    device = None
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    if not device:
        device = torch.device("cpu")

    # Setup dataset and model, with highly variable genes
    adata = sc.read_h5ad(TEST_DATA_PATH)
    n_genes = 10
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
    dataset = MarkovChainDataset(
        adata,
        num_chains=num_chains,
        mask_var="highly_variable",
    )
    match model_type:
        case "ode":
            model = GRNModel(n_genes=dataset.n_genes())
        case "sde":
            model = GRNModelSDE(
                n_genes=dataset.n_genes(),
                shared_sigma=shared_sigma,
                fixed_sigma=fixed_sigma,
                constant_noise=constant_noise,
            )
    model.to(device)

    adj_pre = model.adj.clone().detach()

    y = dataset[:].to(device)
    y_pred = integrate_model(
        y[:, 0, :],
        model,
        n_steps=y.shape[1],
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    untrained_loss = qqloss(y_pred, y.permute(1, 0, 2), dim=1, metric=loss_metric)

    optim = torch.optim.Adam(model.parameters())
    _, _ = train_integration(
        dataset,
        model,
        optim,
        qqloss,
        batch_size=batch_size,
        num_epochs=5 if not batch_size else 2,
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
        loss_kwargs={"metric": loss_metric, "dim": 1},
    )

    y_pred = integrate_model(
        y[:, 1, :],
        model,
        n_steps=y.shape[1],
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    trained_loss = qqloss(y_pred, y.permute(1, 0, 2), dim=1, metric=loss_metric)

    # Weak checks that training did something
    assert trained_loss <= untrained_loss, (
        f"Training should reduce the loss; before: {untrained_loss}, after: {trained_loss}."
    )
    assert torch.any(model.adj != adj_pre), (
        "Training should change the adjacency matrix."
    )
