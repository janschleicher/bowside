"""Module for training of dynamic GRN model."""

import logging
import operator
from math import floor, log10
from pathlib import Path
from typing import Any, List, Literal, Optional, Protocol, Tuple, Union

import equinox as eqx
import jax
import numpy as np
import optax
import torch
from jax import numpy as jnp
from ott.geometry import costs, epsilon_scheduler, pointcloud
from ott.tools import sinkhorn_divergence
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from bowside.dataset import (
    MarkovChainDataset,
    PseudotimeDataset,
    PseudotimeTimepointWrapper,
    split_by_time_point,
)
from bowside.model import GRNModel, GRNModelSDE

# Shim for ott-jax compatibility with JAX >= 0.9
if not hasattr(jax.interpreters.batching, "is_vmappable"):  # type: ignore
    jax.interpreters.batching.is_vmappable = lambda x: isinstance(  # type: ignore
        x,
        (jax.core.Tracer, jax.Array),  # type: ignore
    )


class EarlyStopping:
    """
    Early stopping during model training.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait for an improvement before stopping.
        Default: 5.
    delta : float, optional
        Minimum difference between consecutive loss values to consider
        as an improvement. Default: 0.0.
    mode : Literal["min", "max"], optional
        The direction of the optimization. In "min" mode, training will
        stop when the monitored quantity stops decreasing, in "max" mode
        when it stops increasing. Default: "min".
    logger : Optional[logging.Logger], optional
        Logger instance for logging output. Default: None (uses logger
        of this file).
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        logger: Optional[logging.Logger] = None,
    ):
        # Value checks
        if not isinstance(patience, int):
            raise TypeError(
                f"Parameter `patience` should be int, was {type(patience)}."
            )
        elif patience <= 0:
            raise ValueError("Parameter `patience` must be strictly positive.")

        if not isinstance(delta, float):
            raise TypeError(f"Parameter `delta` should be float, was {type(delta)}.")
        elif delta < 0:
            raise ValueError("Parameter `delta` cannot be negative.")

        if mode not in ["min", "max"]:
            raise ValueError("Parameter `mode` must be one of ['min', 'max'].")

        # Set attributes
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_value = jnp.inf
        if self.mode == "max":
            self.best_value = -self.best_value
            self.compare = operator.gt
        else:
            self.compare = operator.lt
        self.epochs_without_improvement = 0

        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def should_stop(self, value: float) -> bool:
        """
        Check whether model training should be stopped.

        Parameters
        ----------
        value : float
            Current value of the quantity tracked for early stopping.

        Returns
        -------
        bool
            Whether to stop training.
        """
        comparison_value = self.best_value
        if self.mode == "max":
            comparison_value += self.delta
        else:
            comparison_value -= self.delta

        if self.compare(value, comparison_value):
            self.best_value = value
            self.epochs_without_improvement = 0
            self.logger.debug(
                "Model improved, resetting number of epochs without improvement."
            )
            return False
        else:
            self.epochs_without_improvement += 1
            self.logger.debug("No model improvement.")
            if self.epochs_without_improvement >= self.patience:
                self.logger.info("Early stopping triggered.")
                return True
        return False


@jax.jit
def mse_loss(
    y_pred: jax.Array, y_true: jax.Array, weight: Optional[jax.Array] = None
) -> jax.Array:
    """
    Compute the mean squared error between two arrays.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values.
    y_true : jax.Array
        True values.
    weight : Optional[jax.Array], optional
        Weights for each sample. Default: None.

    Returns
    -------
    jax.Array
        Mean squared error between arrays.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Incompatible shapes of `y_pred`: {y_pred.shape} and `y_true`: {y_true.shape}."
        )
    if weight is not None:
        if weight.shape != y_pred.shape:
            raise ValueError(
                f"Incompatible shapes of `weight`: {weight.shape} and `y_pred`: {y_pred.shape}."
            )
        return jnp.sum(jnp.pow(y_pred - y_true, 2) * weight) / jnp.sum(weight)
    return jnp.mean(jnp.pow(y_pred - y_true, 2))


@jax.jit
def mae_loss(
    y_pred: jax.Array, y_true: jax.Array, weight: Optional[jax.Array] = None
) -> jax.Array:
    """
    Compute the mean absolute error between two arrays.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values.
    y_true : jax.Array
        True values.
    weight : Optional[jax.Array], optional
        Weights for each sample. Default: None.

    Returns
    -------
    jax.Array
        Mean absolute error between arrays.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Incompatible shapes of `y_pred`: {y_pred.shape} and `y_true`: {y_true.shape}."
        )
    if weight is not None:
        if weight.shape != y_pred.shape:
            raise ValueError(
                f"Incompatible shapes of `weight`: {weight.shape} and `y_pred`: {y_pred.shape}."
            )
        return jnp.sum(jnp.abs(y_pred - y_true) * weight) / jnp.sum(weight)
    return jnp.mean(jnp.abs(y_pred - y_true))


class LossFunction(Protocol):
    def __call__(
        self,
        y_pred: jax.Array,
        y_true: jax.Array,
        weight: Optional[jax.Array] = None,
        *args: Any,
        **kwargs: Any,
    ) -> jax.Array: ...


@eqx.filter_jit
def qqloss(
    y_pred: jax.Array,
    y_true: jax.Array,
    weight: Optional[jax.Array] = None,
    dim: int = 0,
    n_quantiles: int = 100,
    metric: LossFunction = mse_loss,
) -> jax.Array:
    """
    Compute the difference between the quantiles of two arrays along a
    given dimension.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values.
    y_true : jax.Array
        True values.
    weight : Optional[jax.Array], optional
        Weights for each sample. Only used when `metric` supports this.
        Default: None.
    dim : int, optional
        The dimension to reduce. Default: 0.
    n_quantiles : int, optional
        Number of quantiles. Default: 100.
    metric : LossFunction, optional
        Metric for comparing the quantiles. Default:
        `bowside.training.functional.mse_loss`.

    Returns
    -------
    jax.Array
        Difference between quantiles.
    """
    quantiles = jnp.linspace(0, 1, n_quantiles + 1)
    pred = jnp.quantile(y_pred, quantiles, axis=dim)
    if weight is not None:
        weight = jnp.broadcast_to(weight[None, :, None], pred.shape)
    return metric(pred, jnp.quantile(y_true, quantiles, axis=dim), weight=weight)


@jax.jit(static_argnames=["batch_size"])
def sinkhorn_loss(
    y_pred: jax.Array,
    y_true: jax.Array,
    p: int = 2,
    batch_size: Optional[int] = None,
    blur: float = 0.1,
    decay: float = 0.7,
    weight: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Compute the Sinkhorn divergence between two arrays.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values of shape (n_simulations, n_timepoints, n_var).
    y_true : jax.Array
        True values of shape (n_simulations, n_timepoints, n_var).
    p : int, optional
        Specifies the ground cost function between points. Default: 2.
        Supported values:
        - p=1: Euclidean distance.
        - p=2: Squared Euclidean distance.
        Other powers of the Euclidean distance for p in [1, +\\infty)
        are possible, but not recommended.
    batch_size : Optional[int], optional
        If `batch_size` is set to a positive integer, the online version
        of the Sinkhorn divergence is used, computing the cost matrix on
        the fly for `batch_size` points at a time instead of storing it
        in memory. Otherwise, the complete cost matrix is stored in
        memory. Default: None.
    blur : float, optional
        Entropic regularization strength. Epsilon is calculated as
        blur to the power of `p`. Default: 0.1.
    decay : float, optional
        Ratio between successive values in the epsilon schedule.
        Default: 0.7.
    weight : Optional[jax.Array], optional
        Weights for each sample. Default: None.

    Returns
    -------
    jax.Array
        Average Sinkhorn divergence across samples.
    """
    # Check inputs
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Incompatible shapes of `y_pred`: {y_pred.shape} and `y_true`: {y_true.shape}."
        )
    if y_pred.ndim != 3:
        raise ValueError("Expected shape (n_simulations, n_timepoints, n_var).")

    cost_fn = costs.EuclideanP(p)

    epsilon = blur**p

    def sinkhorn_loss_one(x: jax.Array, y: jax.Array):
        div, _ = sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud,
            x,
            y,
            epsilon=epsilon_scheduler.Epsilon(target=epsilon, decay=decay),
            cost_fn=cost_fn,
            batch_size=batch_size,
        )
        return div

    loss = jax.vmap(sinkhorn_loss_one, in_axes=1)(y_pred, y_true)

    if weight is None:
        return jnp.mean(loss)
    else:
        if weight.shape != loss.shape:
            raise ValueError(
                f"Incompatible shapes of `weight`: {weight.shape} and `loss`: {loss.shape}."
            )
        return jnp.sum(loss * weight) / jnp.sum(weight)


@eqx.filter_jit
def integrate_model(
    model: GRNModel,
    y0: jax.Array,
    time_points: jax.Array,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """
    Integrate a gene regulatory differential equation model (ODE or SDE).

    Parameters
    ----------
    model : GRNModel
        The gene regulatory network model.
    y0 : jax.Array
        Initial state of the differential equation of shape (batch_size, n_var).
    time_points : jax.Array
        Time points to use for integration.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.

    Returns
    -------
    jax.Array
        State array of size (n_simulations, n_steps, n_var).

    Raises
    ------
    TypeError
        An error is raised if the model is of the wrong type.
    ValueError
        An error is raised if no random key is passed when the model is
        a GRNModelSDE.
    """
    if isinstance(model, GRNModelSDE):
        if not isinstance(key, jax.Array):
            raise ValueError(
                "Integration of a SDE model requires a key for the random number generator."
            )
        return jax.vmap(model, in_axes=[None, 0, 0])(
            time_points, y0, jax.random.split(key, y0.shape[0])
        )  # type: ignore
    elif isinstance(model, GRNModel):
        return jax.vmap(model, in_axes=[None, 0])(time_points, y0)  # type: ignore
    else:
        raise TypeError(f"Expected type GRNModel, but got {type(model)}.")


def _raise_invalid_regularization(regularization: Literal["l1", "l2"]):
    valid_regularizations = ["l1", "l2"]
    if regularization not in valid_regularizations:
        raise ValueError(
            f"Invalid regularization option {regularization}. Must be one of {valid_regularizations}"
        )


@eqx.filter_jit
def compute_loss(
    model: GRNModel,
    y: jax.Array,
    time_points: jax.Array,
    loss_func: LossFunction,
    loss_weights: Optional[jax.Array] = None,
    regularization: Optional[Literal["l1", "l2"]] = None,
    regularization_strength: float = 1e-4,
    key: Optional[jax.Array] = None,
    loss_kwargs: dict = {},
) -> jax.Array:
    """
    Compute the loss based on the true data. For this, integration
    starting from y0 is performed first to obtain the predicted data.

    Parameters
    ----------
    model : GRNModel
        The gene regulatory network model.
    y : jax.Array
        True sequences of shape (batch_size, n_steps, n_var).
    time_points : jax.Array
        Time points to use for integration.
    loss_func : LossFunction
        Loss function to optimize.
    loss_weights : Optional[jax.Array], optional
        Weights for each time point/step. Default: None.
    regularization : Optional[Literal["l1", "l2"]], optional
        Regularization method for the adjacency method. Options are "l1"
        and "l2". With the default, None, no regularization is applied.
    regularization_strength : float, optional
        Relative weight of the regularization term added to the loss if
        `regularization` is not None. Default: 1e-4.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    jax.Array
        Value of the loss function.
    """
    y_pred = integrate_model(model, y[:, 0, :], time_points, key)
    loss = loss_func(y_pred, y, weight=loss_weights, **loss_kwargs)
    if regularization is not None:
        jax.debug.callback(_raise_invalid_regularization, regularization)
        reg_idx = {"l1": 0, "l2": 1}
        loss = jax.lax.switch(
            reg_idx[regularization],
            [
                lambda x: x
                + regularization_strength
                * jnp.sum(jnp.abs(model.adj * model.adj_mask)),
                lambda x: x
                + regularization_strength
                * jnp.sum(jnp.pow(model.adj * model.adj_mask, 2)),
            ],
            loss,
        )
    return loss


@eqx.filter_jit
def train_step_integration(
    model: GRNModel,
    y: jax.Array,
    time_points: jax.Array,
    optimizer: optax.GradientTransformationExtraArgs,
    opt_state: optax.OptState,
    loss_func: LossFunction,
    loss_weights: Optional[jax.Array] = None,
    regularization: Optional[Literal["l1", "l2"]] = None,
    regularization_strength: float = 1e-4,
    key: Optional[jax.Array] = None,
    loss_kwargs: dict = {},
) -> Tuple[jax.Array, GRNModel, optax.OptState]:
    """
    A single training step for a gene regulatory network ODE/SDE model,
    including  integration, loss evaluation, backpropagation, and weight
    updates.

    Parameters
    ----------
    model : GRNModel
        The gene regulatory network model.
    y : jax.Array
        Training sequences of shape (batch_size, n_steps, n_var).
    time_points : jax.Array
        Time points to use for integration.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer for model training.
    opt_state : optax.OptState
        Current state of the optimizer.
    loss_func : LossFunction
        Loss function to optimize.
    loss_weights : Optional[jax.Array], optional
        Weights for each time point/step. Default: None.
    regularization : Optional[Literal["l1", "l2"]], optional
        Regularization method for the adjacency method. Options are "l1"
        and "l2". With the default, None, no regularization is applied.
    regularization_strength : float, optional
        Relative weight of the regularization term added to the loss if
        `regularization` is not None. Default: 1e-4.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    Tuple[jax.Array, GRNModel, optax.OptState]
        Training loss after the training step, updated model, and state
        of the optimizer.
    """
    loss, grads = eqx.filter_value_and_grad(compute_loss)(
        model,
        y,
        time_points,
        loss_func,
        loss_weights=loss_weights,
        regularization=regularization,
        regularization_strength=regularization_strength,
        key=key,
        loss_kwargs=loss_kwargs,
    )

    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state


@eqx.filter_jit
def test(
    model: GRNModel,
    y: jax.Array,
    time_points: jax.Array,
    loss_func: LossFunction,
    key: Optional[jax.Array] = None,
    loss_weights: Optional[jax.Array] = None,
    loss_kwargs: dict = {},
) -> jax.Array:
    """
    Evaluate a gene regulatory network ODE/SDE model on a set of test
    sequences by integration.

    Parameters
    ----------
    model : GRNModel
        The gene regulatory network model.
    y : jax.Array
        Test sequences of shape (batch_size, n_steps, n_var).
    time_points : jax.Array
        Time points to use for integration.
    loss_func : LossFunction
        Loss function to optimize.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
    loss_weights : Optional[jax.Array], optional
        Weights for each time point/step. Default: None.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    jax.Array
        Test lost.
    """
    loss = compute_loss(
        model,
        y,
        time_points,
        loss_func,
        loss_weights=loss_weights,
        regularization=None,
        key=key,
        loss_kwargs=loss_kwargs,
    )
    return loss


def train_integration(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    optimizer: optax.GradientTransformationExtraArgs,
    loss_func: LossFunction,
    key: jax.Array,
    loss_weights: Optional[jax.Array] = None,
    regularization: Optional[Literal["l1", "l2"]] = None,
    regularization_strength: float = 1e-4,
    pruning_frequency: Optional[int] = 50,
    pruning_atol: float = 1e-3,
    pruning_rtol: float = 1e-3,
    train_proportion: float = 0.8,
    test_proportion: float = 0.0,
    time_point_split: bool = False,
    batch_size: Optional[int] = None,
    num_epochs: int = 1000,
    early_stopping: bool = True,
    patience: int = 5,
    delta: float = 0.0,
    use_accelerator: bool = True,
    loss_kwargs: dict = {},
    num_workers_loader: int = 0,
    checkpoint: Optional[Union[int, Literal["best", "final"]]] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[GRNModel, List[float], List[float], optax.OptState]:
    """
    Train a gene regulatory network ODE/SDE model using integration.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for training.
    model : GRNModel
        The gene regulatory network model.
    optimizer : optax.GradientTransformationExtraArgs
        Optax optimizer for model training.
    loss_func : LossFunction
        Loss function to optimize.
    key : jax.Array
        Key for the random number generator.
    loss_weights : Optional[jax.Array], optional
        Weights for each time point/step. Default: None.
    regularization : Optional[Literal["l1", "l2"]], optional
        Regularization method for the adjacency method. Options are "l1"
        and "l2". With the default, None, no regularization is applied.
    regularization_strength : float, optional
        Relative weight of the regularization term added to the loss if
        `regularization` is not None. Default: 0.1.
    pruning_frequency : Optional[int], optional
        If `regularization` is not None, remove edges with weight close
        to 0 every `pruning_frequency` epochs. When set to None, edges
        will not be pruned. Default: 50.
    pruning_atol : float, optional
        Absolute threshold for pruning edges close to 0. Default: 1e-3.
    pruning_rtol : float, optional
        Relative threshold for pruning edges close to 0. Default: 1e-3.
    train_proportion : float, optional
        Proportion of samples to use for training. Default: 0.8.
    test_proportion : float, optional
        Proportion of time points to use for testing of interpolation
        performance. Only applicable if `time_point_split` is True. The
        evaluation on the test time points is not done inside this
        training loop. Default: 0.0.
    time_point_split : bool, optional
        Whether to split into training, validation, and optionally test
        data by time point instead of by sample. Default: False.
    batch_size : Optional[int], optional
        Batch size. With the default None, all samples are used at once.
    num_epochs : int, optional
        Number of training epochs. Default: 1000.
    early_stopping : bool, optional
        Whether to stop training when the loss does not improve.
        Default: True.
    patience : int, optional
        Number of epochs to wait for an improvement before stopping.
        Default: 5.
    delta : float, optional
        Minimum difference between consecutive loss values to consider
        as an improvement. Default: 0.0.
    use_accelerator : bool, optional
        Whether to use an accelerator (e.g., CUDA) if it is available.
        Default: True.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.
    num_workers_loader : int, optional
        Number of subprocesses to use for data loading. With the default,
        0, data is loaded in the main process.
    checkpoint : Optional[Union[int, Literal["best", "final"]]], optional
        Save the model state every `checkpoint` epochs, or only save the
        state of the best/final model with "best"/"final". The default
        None corresponds to not saving any checkpoints.
    checkpoint_dir : Optional[Union[str, Path]], optional
        Directory for saving checkpoints. Default: None (will save
        checkpoints to the directory ./checkpoints if `checkpoint` is not
        None).
    logger : Optional[logging.Logger], optional
        Logger instance for logging output. Default: None (uses logger
        of this file).

    Returns
    -------
    Tuple[GRNModel, List[float], List[float], optax.OptState]
        Trained model, train and validation losses for every epoch, and
        final state of the optimizer.

    Raises
    ------
    TypeError
        An error is raised if `checkpoint` is not of type int or if
        `checkpoint_dir` is not of type str or Path.
    ValueError
        An error is raised if `checkpoint` is larger than `num_epochs`.
    NotImplementedError
        An error is raised if splitting by time point is used with a
        :class:`MarkovChainDataset`.
    """
    # Use logger passed to the function if applicable
    if logger is not None:
        log = logger
    else:
        log = logging.getLogger(__name__)
    if checkpoint_dir is None:
        checkpoint_dir = Path("./checkpoints")
    elif isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    elif not isinstance(checkpoint_dir, Path):
        raise TypeError(
            f"Argument `checkpoint_dir` must be of type str or Path, "
            f"but got {type(checkpoint_dir)}"
        )

    if checkpoint is not None:
        if not isinstance(checkpoint, int):
            if checkpoint in ["best", "final"]:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"Saving {checkpoint} model state to directory {checkpoint_dir}."
                )
            else:
                raise TypeError(
                    f"Argument `checkpoint` must be of type int or 'final', but got {type(checkpoint)}."
                )
        elif checkpoint > num_epochs:
            raise ValueError(
                f"Argument `checkpoint` must be smaller than `num_epochs`. "
                f"Got checkpoint={checkpoint}, num_epochs={num_epochs}."
            )
        else:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            log.info(
                f"Saving checkpoints every {checkpoint} epochs to directory {checkpoint_dir}."
            )

    if early_stopping:
        early_stopping_instance = EarlyStopping(patience, delta, "min", log)

    log.info(f"Using model of type {model.__class__.__name__}")

    # Select GPU if available
    device = jax.devices()[0]
    if str(device.platform) != "cpu" and not use_accelerator:
        device = jax.devices("cpu")[0]
        model = jax.device_get(model)
    log.info(f"Running on device '{device}'")

    time_points = None
    if isinstance(dataset, PseudotimeDataset):
        time_points = jnp.asarray(dataset.time_points.numpy().copy())
    elif isinstance(dataset, MarkovChainDataset):
        n_steps = dataset.markov_chains.shape[1]
        time_points = jnp.linspace(0, n_steps - 1, n_steps)
    else:
        raise TypeError(
            "Incompatible dataset type. Use `MarkovChainDataset` or `PseudotimeDataset`."
        )

    if regularization is not None:
        if regularization not in ["l1", "l2"]:
            raise ValueError(
                "Invalid regularization option. Must be one of ['l1', 'l2']."
            )
    if not isinstance(regularization_strength, float):
        raise TypeError("Parameter `regularization_strength` must be of type float.")
    if regularization_strength < 0:
        raise ValueError("Parameter `regularization_strength` must be positive.")
    if pruning_frequency is not None:
        if not isinstance(pruning_frequency, int):
            raise TypeError("Parameter `pruning_frequency` must be of type int.")
        if pruning_frequency <= 0:
            raise ValueError("Parameter `pruning_frequency` must be positive.")

    if loss_weights is not None:
        if loss_weights.shape != (len(time_points),):
            raise ValueError(
                f"Incorrect shape of the loss weights: expected: ({len(time_points)},), actual: {loss_weights.shape}"
            )

    # Split the dataset into training and validation
    if time_point_split:
        if isinstance(dataset, MarkovChainDataset):
            raise NotImplementedError(
                "Splitting by time point not implemented for MarkovChainDataset."
            )
        validation_proportion = 1.0 - train_proportion - test_proportion
        indices = split_by_time_point(
            dataset,
            train_proportion=train_proportion,
            validation_proportion=validation_proportion,
            test_proportion=test_proportion,
        )
        if len(indices) == 2:
            train_indices, validation_indices = indices
        else:
            train_indices, validation_indices, test_indices = indices
            log.info(
                f"Indices of test time points (not used here, use externally "
                f"to test interpolation performance): {test_indices}"
            )
        log.info(f"Indices of training time points: {train_indices}")
        log.info(f"Indices of validation time points: {validation_indices}")

        # Add t0 to validation indices to facilitate integration from y0
        validation_indices = torch.cat(
            [torch.tensor([0], dtype=torch.int), validation_indices]
        )
        trainset = PseudotimeTimepointWrapper(dataset, train_indices)
        valset = PseudotimeTimepointWrapper(dataset, validation_indices)
        train_time_points = jnp.asarray(trainset.time_points.numpy())
        val_time_points = jnp.asarray(valset.time_points.numpy())

        train_weights = None
        val_weights = jnp.ones(len(val_time_points)).at[0].set(0)
        if loss_weights is not None:
            train_weights = loss_weights[jnp.asarray(train_indices.numpy())]
            # Ensure that y0 is not used for calculating the validation loss
            val_weights = (
                loss_weights[jnp.asarray(validation_indices.numpy())].at[0].set(0)
            )

    else:
        if test_proportion > 0.0:
            log.warning(
                "For splitting by sample, a separate test set is not implemented. "
                "If this is required, the dataset should be split by cells before training."
            )
        generator = torch.Generator().manual_seed(42)
        train_length = floor(len(dataset) * train_proportion)
        trainset, valset = torch.utils.data.random_split(
            dataset, [train_length, len(dataset) - train_length], generator=generator
        )
        train_weights = loss_weights
        val_weights = loss_weights
        train_time_points = time_points
        val_time_points = time_points

    # No batch size given: Use the whole dataset
    if batch_size is not None:
        batch_size_train = batch_size
        batch_size_test = batch_size
    else:
        batch_size_train = len(trainset)
        batch_size_test = len(valset)
    log.info(
        f"Batch size train: {batch_size_train}; batch size test: {batch_size_test}"
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers_loader,
    )
    testloader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=num_workers_loader,
    )

    # Train for multiple epochs
    num_batches_train = len(trainloader)
    num_batches_test = len(testloader)
    train_losses: List[float] = []
    val_losses: List[float] = []
    log.info(
        f"Number of training batches: {num_batches_train}; number of test batches: {num_batches_test}"
    )

    log.info("Starting GRN model training.")

    best_model_epoch = 0
    best_test_loss = jnp.inf
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    with logging_redirect_tqdm():
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            # Training
            train_loss = 0.0
            for data in trainloader:
                key, sub_key = jax.random.split(key)
                y = jnp.asarray(data.numpy().copy(), device=device)
                loss, model, opt_state = train_step_integration(
                    model=model,
                    y=y,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    loss_func=loss_func,
                    loss_weights=train_weights,
                    regularization=regularization,
                    regularization_strength=regularization_strength,
                    time_points=train_time_points,
                    key=sub_key,
                    loss_kwargs=loss_kwargs,
                )
                train_loss += loss.item()
            train_loss = train_loss / num_batches_train
            train_losses.append(train_loss)

            # Evaluation
            val_loss = 0.0
            for data in testloader:
                key, sub_key = jax.random.split(key)
                y = jnp.asarray(data.numpy().copy(), device=device)
                loss = test(
                    model=model,
                    y=y,
                    time_points=val_time_points,
                    loss_func=loss_func,
                    key=sub_key,
                    loss_weights=val_weights,
                    loss_kwargs=loss_kwargs,
                )
                val_loss += loss.item()
            val_loss = val_loss / num_batches_test
            val_losses.append(val_loss)

            log.debug(
                f"Epoch {epoch}: training loss: {train_loss}, test loss: {val_loss}"
            )

            # Prune edges close to 0
            if regularization is not None and pruning_frequency is not None:
                if epoch % pruning_frequency == 0 and epoch > 0:
                    log.debug(f"Pruning at epoch {epoch}.")
                    model = model.prune(atol=pruning_atol, rtol=pruning_rtol)

            # Save checkpoint
            if isinstance(checkpoint, int) and epoch % checkpoint == 0:
                model.save(
                    checkpoint_dir
                    / f"checkpoint_e{epoch:0{floor(log10(num_epochs)) + 1}d}.eqx"
                )
            elif checkpoint == "best":
                if val_loss < best_test_loss:
                    model.save(checkpoint_dir / "checkpoint_best_model.eqx")
                    best_model_epoch = epoch

            # Check early stopping
            if early_stopping:
                if early_stopping_instance.should_stop(val_loss):
                    break

        log.info("Finished model training.")

        # Prune edges close to 0
        if regularization is not None and pruning_frequency is not None:
            log.info("Final pruning step of edges with weight close to 0.")
            model = model.prune(atol=pruning_atol, rtol=pruning_rtol)

        # Save final model
        if checkpoint is not None:
            model.save(checkpoint_dir / "checkpoint_final_model.eqx")
            np.save(checkpoint_dir / "train_losses.npy", np.asarray(train_losses))
            np.save(checkpoint_dir / "test_losses.npy", np.asarray(val_losses))
            if checkpoint == "best":
                log.info(
                    f"Best model was at epoch {best_model_epoch}, saved "
                    f"as {str(checkpoint_dir / 'checkpoint_best_model.eqx')}"
                )

        return model, train_losses, val_losses, opt_state
