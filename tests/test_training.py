import os
import shutil
from functools import partial
from math import floor, log10
from pathlib import Path
from typing import Literal, Optional, Union

import equinox as eqx
import jax
import numpy as np
import optax
import pytest
import scanpy as sc
import scvelo as scv
import torch
from jax import numpy as jnp

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.model import GRNModel, GRNModelSDE
from bowside.training import (
    EarlyStopping,
    LossFunction,
    integrate_model,
    mae_loss,
    mse_loss,
    qqloss,
    sinkhorn_loss,
    train_integration,
)
from tests.conftest import TEST_DATA_PATH

key = jax.random.key(42)
key_perfect, key_two_off, key_true_small, key_pred_small = jax.random.split(key, 4)

Y_TRUE = jnp.arange(1, 101, dtype=float).reshape((10, 10))
Y_PRED_PERFECT = Y_TRUE[:, jax.random.permutation(key_perfect, 10)]
Y_PRED_TWO_OFF = (Y_TRUE + 2)[:, jax.random.permutation(key_two_off, 10)]
Y_TRUE_SMALL = jnp.asarray([[1, 1, 1, 1, 1, 3, 8, 8, 8, 8, 8]], dtype=float)[
    :, jax.random.permutation(key_true_small, 11)
]
Y_PRED_SMALL = jnp.asarray([[3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6]], dtype=float)[
    :, jax.random.permutation(key_pred_small, 11)
]

LOSSES_MIN = jnp.hstack([np.linspace(1.0, 0.1, 10), np.ones(20) * 0.1])
LOSSES_MIN_DELTA = jnp.hstack([np.linspace(1.0, 0.1, 10), np.linspace(0.095, 0.0, 20)])
LOSSES_MAX = jnp.hstack([np.linspace(0.0, 0.9, 10), np.ones(20) * 0.9])
LOSSES_MAX_DELTA = jnp.hstack([np.linspace(0.0, 0.9, 10), np.linspace(0.905, 1.0, 20)])

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
    global Y_TRUE_PT, Y_PRED_PT
    Y_TRUE_PT = jnp.asarray(dataset[:100].detach().cpu().numpy())
    Y_PRED_PT = Y_TRUE_PT[
        jax.random.permutation(jax.random.key(42), Y_TRUE_PT.shape[0])
    ]


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
        (5, 0.099, "max", LOSSES_MAX_DELTA, 15),
    ],
)
def test_early_stopping(
    patience: int,
    delta: float,
    mode: Literal["min", "max"],
    losses: jax.Array,
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
        assert expected_epochs < losses.shape[0], (
            "The early stopping should have been triggered."
        )
    else:
        assert expected_epochs == losses.shape[0], (
            "Training should have proceeded to the end."
        )


@pytest.mark.parametrize(
    "y_pred, y_true, metric, expected_loss",
    [
        [Y_PRED_PERFECT, Y_TRUE, mae_loss, 0],
        [Y_PRED_PERFECT, Y_TRUE, mse_loss, 0],
        [Y_PRED_TWO_OFF, Y_TRUE, mae_loss, 2],
        [Y_PRED_TWO_OFF, Y_TRUE, mse_loss, 4],
        [Y_PRED_SMALL, Y_TRUE_SMALL, mae_loss, 2],
        [Y_PRED_SMALL, Y_TRUE_SMALL, mse_loss, 4],
    ],
)
def test_qqloss(
    y_pred: jax.Array,
    y_true: jax.Array,
    metric: LossFunction,
    expected_loss: float,
):
    loss = qqloss(y_pred, y_true, dim=1, n_quantiles=10, metric=metric)
    assert jnp.isclose(loss, expected_loss), (
        f"Wrong loss value. Expected: {expected_loss:.2f}, actual: {loss.item():.2f}"
    )
    assert loss != metric(y_pred, y_true), (
        "The qqloss should not be the same as the underlying metric for the given test cases."
    )


@pytest.mark.parametrize(
    "y_pred, y_true, expected_loss",
    [
        [Y_PRED_PERFECT.T[:, None, :], Y_TRUE.T[:, None, :], 0],
        [Y_PRED_TWO_OFF.T[:, None, :], Y_TRUE.T[:, None, :], None],
        [Y_PRED_SMALL.T[:, None, :], Y_TRUE_SMALL.T[:, None, :], None],
    ],
)
def test_sinkhorn_loss(
    y_pred: jax.Array,
    y_true: jax.Array,
    expected_loss: Optional[float],
):
    loss = sinkhorn_loss(y_pred, y_true)
    if expected_loss is not None:
        assert jnp.isclose(loss, expected_loss, atol=1e-7), (
            f"Wrong loss value. Expected: {expected_loss:.4g}, actual: {loss:.4g}"
        )
    else:
        assert loss > 0, "Loss value should be greater than 0."


@pytest.mark.parametrize("loss", ["qqloss", "sinkhorn"])
@pytest.mark.parametrize("weight", [None, jnp.linspace(0, 10, 20)])
def test_loss_pseudotime_data(loss: Literal["qqloss", "sinkhorn"], weight: jax.Array):
    if loss == "qqloss":
        loss_func = partial(qqloss, dim=0)
    elif loss == "sinkhorn":
        loss_func = sinkhorn_loss
    else:
        raise ValueError("Invalid loss function. Must be one of ['qqloss', 'sinkhorn']")
    loss_val = loss_func(
        Y_PRED_PT,
        Y_TRUE_PT,
        weight=weight,
    )
    assert jnp.isclose(loss_val, 0, atol=1e-4), (
        f"Loss value should be zero, but is {loss_val:.4g}."
    )


@pytest.mark.parametrize(
    "model_type, regularization, batch_size, num_sims, shared_sigma, fixed_sigma, "
    "constant_noise, dataset_type, checkpoint, checkpoint_dir",
    [
        ["ode", None, None, 100, None, None, None, "markov", "final", OUT_DIR],
        ["ode", "l1", None, 100, None, None, None, "markov", "final", OUT_DIR],
        ["ode", "l2", None, 100, None, None, None, "markov", "final", OUT_DIR],
        ["ode", None, None, 100, None, None, None, "markov", "best", OUT_DIR],
        ["ode", None, None, 100, None, None, None, "pseudo", None, None],
        ["ode", None, 100, 500, None, None, None, "markov", 1, "test_out"],
        ["ode", None, 100, 500, None, None, None, "pseudo", 1, None],
        ["sde", None, None, 100, True, True, True, "markov", "final", OUT_DIR],
        ["sde", None, None, 100, True, True, True, "markov", "best", OUT_DIR],
        ["sde", None, None, 100, True, True, False, "markov", None, None],
        ["sde", None, None, 100, True, False, True, "markov", None, None],
        ["sde", None, None, 100, False, False, True, "markov", None, None],
    ],
)
def test_train_integration(
    model_type: Literal["ode", "sde"],
    regularization: Optional[Literal["l1", "l2"]],
    batch_size: int,
    num_sims: int,
    shared_sigma: bool,
    fixed_sigma: bool,
    constant_noise: bool,
    dataset_type: Literal["markov", "pseudo"],
    checkpoint: Optional[Union[int, Literal["best", "final"]]],
    checkpoint_dir: Optional[Path],
):
    torch.manual_seed(42)
    dt = 0.1
    atol = 0.1
    rtol = 0.1
    loss_metric = mse_loss

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
        n_steps = dataset.markov_chains.shape[1]
        time_points = jnp.linspace(0, n_steps - 1, n_steps)
    elif dataset_type == "pseudo":
        dataset = PseudotimeDataset(
            adata,
            num_cells=num_sims,
            mask_var="highly_variable",
            pseudotime_key="velocity_pseudotime",
        )
        time_points = jnp.asarray(dataset.time_points.numpy().copy())
    else:
        raise ValueError("Invalid dataset type. Should be one of ['markov', 'pseudo'].")
    key = jax.random.key(42)
    key, sub_key = jax.random.split(key)
    match model_type:
        case "ode":
            model = GRNModel(
                n_genes=dataset.n_genes(),
                solver_atol=atol,
                solver_rtol=rtol,
                dt=dt,
                key=sub_key,
            )
        case "sde":
            model = GRNModelSDE(
                n_genes=dataset.n_genes(),
                shared_sigma=shared_sigma,
                fixed_sigma=fixed_sigma,
                constant_noise=constant_noise,
                solver_atol=atol,
                solver_rtol=rtol,
                dt=dt,
                key=sub_key,
            )

    adj_pre = model.adj.copy()
    y = jnp.asarray(dataset[:].numpy().copy())
    key, sub_key = jax.random.split(key)
    y_pred = integrate_model(model, y[:, 0, :], time_points=time_points, key=sub_key)
    untrained_loss = qqloss(y_pred, y, dim=0, metric=loss_metric)

    optim = optax.adam(learning_rate=0.001)
    if not batch_size:
        num_epochs = 5
    else:
        num_epochs = 2
    key, sub_key = jax.random.split(key)
    model, _, _, _ = train_integration(
        dataset,
        model,
        optim,
        qqloss,
        sub_key,
        regularization=regularization,
        batch_size=batch_size,
        num_epochs=num_epochs,
        loss_kwargs={"metric": loss_metric, "dim": 0},
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
    )

    key, sub_key = jax.random.split(key)
    y_pred = integrate_model(model, y[:, 0, :], time_points=time_points, key=sub_key)
    trained_loss = qqloss(y_pred, y, dim=0, metric=loss_metric)

    # Weak checks that training did something
    assert trained_loss <= untrained_loss, (
        f"Training should reduce the loss; before: {untrained_loss}, after: {trained_loss}."
    )
    assert jnp.any(model.adj != adj_pre), "Training should change the adjacency matrix."

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
                    / f"checkpoint_e{i:0{floor(log10(num_epochs)) + 1}d}.eqx"
                ), f"Checkpoint for epoch {i} is missing."
        assert os.path.isfile(checkpoint_dir / "checkpoint_final_model.eqx"), (
            "Final model was not saved."
        )
        if checkpoint == "best":
            assert os.path.isfile(checkpoint_dir / "checkpoint_best_model.eqx"), (
                "Best model was not saved."
            )
        shutil.rmtree(checkpoint_dir)


def test_training_exceptions():
    key = jax.random.key(42)
    key, sub_key = jax.random.split(key)
    model = GRNModel(n_genes=dataset.n_genes(), key=sub_key)
    optimizer = optax.adam(learning_rate=0.001)
    train_func = partial(
        train_integration,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        loss_func=sinkhorn_loss,
        key=sub_key,
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
    wrong_model = eqx.nn.MLP(5, 5, 5, 5, key=sub_key)
    with pytest.raises(TypeError, match="Expected type GRNModel, but got "):
        integrate_model(wrong_model, y0, jnp.ones(10))  # type: ignore

    # Wrong dataset type
    with pytest.raises(
        TypeError,
        match="Incompatible dataset type. Use `MarkovChainDataset` or `PseudotimeDataset`.",
    ):
        train_integration(ADATA, model, optimizer, sinkhorn_loss, sub_key)  # type: ignore

    # Wrong parameters for early stopping
    with pytest.raises(TypeError, match="Parameter `patience` should be int, was"):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            key=sub_key,
            patience=0.2,  # type: ignore
        )
    with pytest.raises(
        ValueError, match="Parameter `patience` must be strictly positive."
    ):
        train_integration(
            dataset, model, optimizer, sinkhorn_loss, key=sub_key, patience=-7
        )
    with pytest.raises(TypeError, match="Parameter `delta` should be float, was"):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            key=sub_key,
            delta="wrong",  # type: ignore
        )
    with pytest.raises(ValueError, match="Parameter `delta` cannot be negative."):
        train_integration(
            dataset, model, optimizer, sinkhorn_loss, key=sub_key, delta=-0.1
        )

    # Wrong regularization
    with pytest.raises(
        ValueError,
        match="Invalid regularization option. Must be one of \\['l1', 'l2'\\]",
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            key=sub_key,
            regularization="l3",  # type: ignore
        )
    with pytest.raises(
        TypeError, match="Parameter `regularization_strength` must be of type float."
    ):
        train_integration(
            dataset,
            model,
            optimizer,
            sinkhorn_loss,
            key=sub_key,
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
            key=sub_key,
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
            key=sub_key,
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
            key=sub_key,
            regularization="l1",
            pruning_frequency=-5,
        )
