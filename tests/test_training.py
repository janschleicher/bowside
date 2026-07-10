import os
import shutil
from collections.abc import Callable
from functools import partial
from math import floor, log10
from pathlib import Path
from typing import Literal, Optional, Union

import pytest
import scanpy as sc
import scvelo as scv
import torch

from bowside.dataset import (
    BifurcatingPseudotimeDataset,
    MarkovChainDataset,
    PseudotimeDataset,
)
from bowside.model import GRNModel, GRNModelSDE
from bowside.training import (
    EarlyStopping,
    integrate_model,
    qqloss,
    sinkhorn_loss,
    train_integration,
)
from tests.conftest import TEST_DATA_PATH

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

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

LOSSES_MIN = torch.hstack([torch.linspace(1.0, 0.1, 10), torch.ones(20) * 0.1])
LOSSES_MIN_DELTA = torch.hstack(
    [torch.linspace(1.0, 0.1, 10), torch.linspace(0.095, 0.0, 20)]
)
LOSSES_MAX = torch.hstack([torch.linspace(0.0, 0.9, 10), torch.ones(20) * 0.9])
LOSSES_MAX_DELTA = torch.hstack(
    [torch.linspace(0.0, 0.9, 10), torch.linspace(0.905, 1.0, 20)]
)

OUT_DIR = Path("test_out/")


@pytest.fixture(scope="module", autouse=True)
def prepare():
    global ADATA
    ADATA = sc.read_h5ad(TEST_DATA_PATH)
    ADATA.obsp["T_forward"] = scv.utils.get_transition_matrix(
        ADATA, self_transitions=False
    ).toarray()
    scv.tl.terminal_states(ADATA, self_transitions=False)
    scv.tl.velocity_pseudotime(ADATA)

    global dataset
    dataset = PseudotimeDataset(
        ADATA, pseudotime_key="velocity_pseudotime", num_cells=500, copy=True
    )
    generator = torch.manual_seed(42)
    global Y_TRUE_PT, Y_PRED_PT
    Y_TRUE_PT = dataset[:].permute(1, 0, 2).contiguous()
    Y_PRED_PT = (
        dataset[:][torch.randperm(len(dataset), generator=generator), :, :]
        .permute(1, 0, 2)
        .contiguous()
    )


@pytest.mark.parametrize(
    "patience, delta, mode, losses, expected_epochs",
    [
        (1, 0.0, "min", LOSSES_MIN, 11),
        (5, 0.0, "min", LOSSES_MIN, 15),
        (10, 0.0, "min", LOSSES_MIN, 20),
        (5, 0.01, "min", LOSSES_MIN, 15),
        (5, 0.0, "min", LOSSES_MIN_DELTA, 30),
        (5, 0.1, "min", LOSSES_MIN_DELTA, 15),
        (5, 0.0, "min", LOSSES_MAX, 6),
        (5, 0.0, "max", LOSSES_MAX, 15),
        (5, 0.0, "max", LOSSES_MAX_DELTA, 30),
        (5, 0.1, "max", LOSSES_MAX_DELTA, 15),
    ],
)
def test_early_stopping(
    patience: int,
    delta: float,
    mode: Literal["min", "max"],
    losses: torch.Tensor,
    expected_epochs: int,
):
    early_stopping = EarlyStopping(patience=patience, delta=delta, mode=mode)
    for epoch, loss_val in enumerate(losses):
        should_stop = early_stopping.should_stop(loss_val.item())
        if should_stop:
            break
    # Increment because iteration starts at 0
    assert epoch + 1 == expected_epochs, (
        f"Training should have stopped after {expected_epochs} epochs, but ended after {epoch + 1}."
    )
    if should_stop:
        assert expected_epochs < losses.size(0), (
            "The early stopping should have been triggered."
        )
    else:
        assert expected_epochs == losses.size(0), (
            "Training should have proceeded to the end."
        )


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
    "y_pred, y_true, expected_loss",
    [
        [Y_PRED_PERFECT.T, Y_TRUE.T, 0],
        [Y_PRED_TWO_OFF, Y_TRUE, None],
        [Y_PRED_SMALL, Y_TRUE_SMALL, None],
    ],
)
def test_sinkhorn_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expected_loss: Optional[float],
):
    loss = sinkhorn_loss(y_pred, y_true)
    if expected_loss is not None:
        assert torch.isclose(loss, torch.tensor(expected_loss, dtype=loss.dtype)), (
            f"Wrong loss value. Expected: {expected_loss:.2f}, actual: {loss.item():.2f}"
        )
    else:
        assert loss > 0, "Loss value should be greater than 0."


@pytest.mark.parametrize("loss", ["qqloss", "sinkhorn"])
@pytest.mark.parametrize("weight", [None, torch.linspace(0, 10, 20)])
def test_loss_pseudotime_data(
    loss: Literal["qqloss", "sinkhorn"], weight: torch.Tensor
):
    if loss == "qqloss":
        loss_func = partial(qqloss, dim=1)
    elif loss == "sinkhorn":
        loss_func = sinkhorn_loss
    else:
        raise ValueError("Invalid loss function. Must be one of ['qqloss', 'sinkhorn']")
    loss_val = loss_func(
        Y_PRED_PT,
        Y_TRUE_PT,
        weight=weight,
    )
    assert torch.isclose(loss_val, torch.tensor(0, dtype=loss_val.dtype), atol=1e-4), (
        f"Loss value should be zero, but is {loss_val.item():.4g}."
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Time limit.")
def test_sinkhorn_loss_bifurcating_sde():
    y0 = torch.zeros((500, 2))
    dataset = BifurcatingPseudotimeDataset(y0)
    y_true = dataset[:].permute(1, 0, 2).contiguous()
    ts = dataset.time_points
    y_pred_true_model = integrate_model(dataset.sde, y0, time_points=ts, dt=0.1)
    loss_true_model = sinkhorn_loss(y_pred_true_model, y_true)
    atol, rtol = 1e-3, 1e-3
    torch_zero = torch.tensor(0, dtype=torch.float32)
    assert torch.isclose(loss_true_model, torch_zero, atol=atol, rtol=rtol), (
        "The loss of the true model should be close to 0."
    )
    model = GRNModelSDE(
        dataset.n_genes(),
        sigma=0.1,
        shared_sigma=True,
        fixed_sigma=True,
        constant_noise=True,
    )
    y_pred_init_model = integrate_model(model, y0, time_points=ts, dt=0.1)
    loss_init_model = sinkhorn_loss(y_pred_init_model, y_true)
    assert loss_init_model > loss_true_model, (
        "The loss of the untrained model should be higher than the one of the true model."
    )
    assert ~torch.isclose(loss_init_model, torch_zero, atol=atol, rtol=rtol), (
        "The loss of the untrained model should not be close to 0."
    )
    assert loss_init_model > 0, (
        "The loss of the untrained model should be greater than 0."
    )


@pytest.mark.parametrize(
    "model_type, loss_weights, time_point_split, regularization,"
    "batch_size, num_sims, shared_sigma, fixed_sigma, constant_noise, "
    "dataset_type, checkpoint, checkpoint_dir, adjoint",
    [
        [
            "ode",
            None,
            False,
            None,
            None,
            100,
            None,
            None,
            None,
            "markov",
            "final",
            OUT_DIR,
            False,
        ],
        [
            "ode",
            None,
            False,
            "l1",
            None,
            100,
            None,
            None,
            None,
            "markov",
            "final",
            OUT_DIR,
            False,
        ],
        [
            "ode",
            None,
            False,
            "l2",
            None,
            100,
            None,
            None,
            None,
            "markov",
            "final",
            OUT_DIR,
            False,
        ],
        [
            "ode",
            None,
            False,
            None,
            None,
            100,
            None,
            None,
            None,
            "markov",
            "best",
            OUT_DIR,
            False,
        ],
        [
            "ode",
            None,
            False,
            None,
            None,
            100,
            None,
            None,
            None,
            "pseudo",
            None,
            None,
            True,
        ],
        [
            "ode",
            None,
            False,
            None,
            100,
            500,
            None,
            None,
            None,
            "markov",
            1,
            "test_out",
            False,
        ],
        [
            "ode",
            None,
            False,
            None,
            100,
            500,
            None,
            None,
            None,
            "pseudo",
            1,
            None,
            False,
        ],
        [
            "sde",
            None,
            False,
            None,
            None,
            100,
            True,
            True,
            True,
            "markov",
            "final",
            OUT_DIR,
            False,
        ],
        [
            "sde",
            None,
            False,
            None,
            None,
            100,
            True,
            True,
            True,
            "markov",
            "best",
            OUT_DIR,
            False,
        ],
        [
            "sde",
            None,
            False,
            None,
            None,
            100,
            True,
            True,
            False,
            "markov",
            None,
            None,
            True,
        ],
        [
            "sde",
            None,
            False,
            None,
            None,
            100,
            True,
            False,
            True,
            "markov",
            None,
            None,
            False,
        ],
        [
            "sde",
            None,
            False,
            None,
            None,
            100,
            False,
            False,
            True,
            "markov",
            None,
            None,
            False,
        ],
        [
            "sde",
            torch.arange(20),
            False,
            None,
            None,
            100,
            False,
            False,
            True,
            "pseudo",
            None,
            None,
            False,
        ],
        [
            "sde",
            torch.arange(20),
            True,
            None,
            None,
            100,
            False,
            False,
            True,
            "pseudo",
            None,
            None,
            False,
        ],
        [
            "sde",
            None,
            True,
            None,
            None,
            100,
            False,
            False,
            True,
            "pseudo",
            None,
            None,
            False,
        ],
    ],
)
def test_train_integration(
    model_type: Literal["ode", "sde"],
    loss_weights: Optional[torch.Tensor],
    time_point_split: bool,
    regularization: Optional[Literal["l1", "l2"]],
    batch_size: int,
    num_sims: int,
    shared_sigma: bool,
    fixed_sigma: bool,
    constant_noise: bool,
    dataset_type: Literal["markov", "pseudo"],
    checkpoint: Optional[Union[int, Literal["best", "final"]]],
    checkpoint_dir: Optional[Path],
    adjoint: bool,
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
    adata = ADATA.copy()
    n_genes = 10
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
    if dataset_type == "markov":
        dataset = MarkovChainDataset(
            adata,
            num_chains=num_sims,
            mask_var="highly_variable",
        )
    elif dataset_type == "pseudo":
        dataset = PseudotimeDataset(
            adata,
            num_cells=num_sims,
            mask_var="highly_variable",
            pseudotime_key="velocity_pseudotime",
        )
    else:
        raise ValueError("Invalid dataset type. Should be one of ['markov', 'pseudo'].")
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
        model,
        y[:, 0, :],
        n_steps=y.shape[1],
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    untrained_loss = qqloss(y_pred, y.permute(1, 0, 2), dim=1, metric=loss_metric)

    optim = torch.optim.Adam(model.parameters())
    if not batch_size:
        num_epochs = 5
    else:
        num_epochs = 2
    _, _ = train_integration(
        dataset,
        model,
        optim,
        qqloss,
        loss_weights=loss_weights,
        time_point_split=time_point_split,
        regularization=regularization,
        batch_size=batch_size,
        num_epochs=num_epochs,
        adjoint=adjoint,
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
        loss_kwargs={"metric": loss_metric, "dim": 1},
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    y_pred = integrate_model(
        model,
        y[:, 1, :],
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

    # Check if model was saved
    if checkpoint is not None:
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        elif isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.exists(), "Checkpoint directory was not created."
        if isinstance(checkpoint, int):
            for i in range(0, num_epochs, checkpoint):
                assert os.path.isfile(
                    checkpoint_dir
                    / f"checkpoint_e{i:0{floor(log10(num_epochs)) + 1}d}.tar"
                ), f"Checkpoint for epoch {i} is missing."
        assert os.path.isfile(checkpoint_dir / "checkpoint_final_model.tar"), (
            "Final model was not saved."
        )
        if checkpoint == "best":
            assert os.path.isfile(checkpoint_dir / "checkpoint_best_model.tar"), (
                "Best model was not saved."
            )
        shutil.rmtree(checkpoint_dir)


@pytest.mark.parametrize(
    "regularization, pruning_frequency",
    [[None, None], ["l1", None], ["l1", 2], ["l2", None]],
)
def test_train_integration_maximization(
    regularization: Optional[Literal["l1", "l2"]], pruning_frequency: Optional[int]
):
    torch.manual_seed(42)
    dt = 0.1
    atol = 0.1
    rtol = 0.1
    solver_method = "adaptive_heun"
    loss_metric = torch.nn.functional.mse_loss
    regularization_strength = 0.1

    # Select GPU if available
    device = None
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    if not device:
        device = torch.device("cpu")

    # Setup dataset and model, with highly variable genes
    adata = ADATA.copy()
    n_genes = 10
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
    dataset = PseudotimeDataset(
        adata,
        num_cells=100,
        mask_var="highly_variable",
        pseudotime_key="velocity_pseudotime",
    )

    model = GRNModel(n_genes=dataset.n_genes())
    model.to(device)

    adj_pre = model.adj.clone().detach()

    y = dataset[:].to(device)
    y_pred = integrate_model(
        model,
        y[:, 0, :],
        n_steps=y.shape[1],
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )

    # Invert loss function to test maximization
    def qqloss_max(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        dim: int = 1,
        n_quantiles: int = 100,
        metric: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = torch.nn.functional.mse_loss,
        weight: Optional[torch.Tensor] = None,
    ):
        return -qqloss(
            y_pred=y_pred,
            y_true=y_true,
            dim=dim,
            n_quantiles=n_quantiles,
            metric=metric,
            weight=weight,
        )

    untrained_loss = qqloss_max(y_pred, y.permute(1, 0, 2), dim=1, metric=loss_metric)
    if regularization is not None:
        if regularization == "l1":
            untrained_loss -= (
                regularization_strength * (model.adj * model.adj_mask).abs().sum()
            )
        elif regularization == "l2":
            untrained_loss -= (
                regularization_strength * (model.adj * model.adj_mask).pow(2).sum()
            )

    optim = torch.optim.Adam(model.parameters(), maximize=True)
    num_epochs = 5

    _, _ = train_integration(
        dataset,
        model,
        optim,
        qqloss_max,
        regularization=regularization,
        regularization_strength=regularization_strength,
        pruning_frequency=pruning_frequency,
        num_epochs=num_epochs,
        optimization_mode="max",
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
        loss_kwargs={"metric": loss_metric, "dim": 1},
        checkpoint="best",
    )

    y_pred = integrate_model(
        model,
        y[:, 1, :],
        n_steps=y.shape[1],
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    trained_loss = qqloss_max(y_pred, y.permute(1, 0, 2), dim=1, metric=loss_metric)
    if regularization is not None:
        if regularization == "l1":
            trained_loss -= (
                regularization_strength * (model.adj * model.adj_mask).abs().sum()
            )
        elif regularization == "l2":
            trained_loss -= (
                regularization_strength * (model.adj * model.adj_mask).pow(2).sum()
            )

    # Weak checks that training did something
    assert trained_loss >= untrained_loss, (
        f"Training should increase the target; before: {untrained_loss}, after: {trained_loss}."
    )
    assert torch.any(model.adj != adj_pre), (
        "Training should change the adjacency matrix."
    )

    # Check if model was saved
    checkpoint_dir = Path("./checkpoints")
    assert checkpoint_dir.exists(), "Checkpoint directory was not created."
    assert os.path.isfile(checkpoint_dir / "checkpoint_best_model.tar"), (
        "Best model was not saved."
    )
    shutil.rmtree(checkpoint_dir)


def test_training_exceptions():
    model = GRNModel(n_genes=dataset.n_genes())
    optimizer = torch.optim.Adam(model.parameters())
    train_func = partial(
        train_integration,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        loss_func=sinkhorn_loss,
    )
    # Wrong checkpoint configuration
    with pytest.raises(
        TypeError, match="Argument `checkpoint_dir` must be of type str or Path"
    ):
        train_func(checkpoint_dir=42)  # type: ignore
    with pytest.raises(
        ValueError, match="Argument `checkpoint` must be smaller than `num_epochs`."
    ):
        train_func(num_epochs=5, checkpoint=10)  # type: ignore
    with pytest.raises(
        TypeError,
        match="Argument `checkpoint` must be of type int or 'final', but got ",
    ):
        train_func(checkpoint="all")  # type: ignore

    # Wrong model
    y0 = dataset[:][:, 0, :]
    wrong_model = torch.nn.Module()
    with pytest.raises(TypeError, match="Expected type GRNModel, but got "):
        integrate_model(y0, wrong_model, 10)  # type: ignore

    # Wrong number of steps/time points configuration
    with pytest.raises(
        ValueError,
        match="Arguments `n_steps` and `time_points` are both `None`. You must set one of them.",
    ):
        integrate_model(model, y0)
    with pytest.raises(
        ValueError,
        match="Arguments `n_steps` and `time_points` are mutually exclusive. Choose one.",
    ):
        integrate_model(model, y0, 10, dataset.time_points)

    # Wrong dataset type
    with pytest.raises(
        TypeError,
        match="Incompatible dataset type. Use `MarkovChainDataset` or `PseudotimeDataset`.",
    ):
        train_integration(ADATA, model, optimizer, sinkhorn_loss)  # type: ignore

    # Wrong parameters for early stopping
    with pytest.raises(TypeError, match="Parameter `patience` should be int, was"):
        train_integration(dataset, model, optimizer, sinkhorn_loss, patience=0.2)  # type: ignore
    with pytest.raises(
        ValueError, match="Parameter `patience` must be strictly positive."
    ):
        train_integration(dataset, model, optimizer, sinkhorn_loss, patience=-7)
    with pytest.raises(TypeError, match="Parameter `delta` should be float, was"):
        train_integration(dataset, model, optimizer, sinkhorn_loss, delta="wrong")  # type: ignore
    with pytest.raises(ValueError, match="Parameter `delta` cannot be negative."):
        train_integration(dataset, model, optimizer, sinkhorn_loss, delta=-0.1)
    with pytest.raises(
        ValueError, match="Parameter `mode` must be one of \\['min', 'max'\\]."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            optimization_mode="loss",  # type: ignore
        )

    # Wrong regularization
    with pytest.raises(
        ValueError,
        match="Invalid regularization option. Must be one of \\['l1', 'l2'\\]",
    ):
        train_integration(dataset, model, optimizer, sinkhorn_loss, regularization="l3")  # type: ignore
    with pytest.raises(
        TypeError, match="Parameter `regularization_strength` must be of type float."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            regularization="l1",
            regularization_strength="1",  # type: ignore
        )
    with pytest.raises(
        ValueError, match="Parameter `regularization_strength` must be positive."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            regularization="l1",
            regularization_strength=-0.1,
        )
    with pytest.raises(
        TypeError, match="Parameter `pruning_frequency` must be of type int."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            regularization="l1",
            pruning_frequency=0.5,  # type: ignore
        )
    with pytest.raises(
        ValueError, match="Parameter `pruning_frequency` must be positive."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            regularization="l1",
            pruning_frequency=-5,
        )
