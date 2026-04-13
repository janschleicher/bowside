"""Module for training of dynamic GRN model."""

import inspect
import logging
import operator
from collections.abc import Callable
from functools import partial
from math import floor, log10
from pathlib import Path
from typing import Any, List, Literal, Optional, Protocol, Tuple, Union

import torch
from geomloss import SamplesLoss
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from sc_markov_ode.dataset import MarkovChainDataset, PseudotimeDataset
from sc_markov_ode.model import GRNModel, GRNModelSDE


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
        self.best_value = torch.tensor(torch.inf)
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
        value : torch.Tensor
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


class NaNTensorError(RuntimeError):
    pass


def qqloss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    dim: int = 0,
    n_quantiles: int = 100,
    metric: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.nn.functional.mse_loss,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the difference between the quantiles of two tensors along a
    given dimension.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted values.
    y_true : torch.Tensor
        True values.
    dim : int, optional
        The dimension to reduce. Default: 0.
    n_quantiles : int, optional
        Number of quantiles. Default: 100.
    metric : Callable, optional
        Metric for comparing the quantiles. Default:
        `torch.nn.functional.mse_loss`.
    weight : Optional[torch.Tensor], optional
        Weights for each sample. Only used when `metric` supports this.
        Default: None.

    Returns
    -------
    torch.Tensor
        Difference between quantiles.
    """
    quantiles = torch.linspace(0, 1, n_quantiles + 1, device=y_true.device)
    signature = inspect.signature(metric)
    has_weight = any(param.name == "weight" for param in signature.parameters.values())
    pred = torch.quantile(y_pred, quantiles, dim=dim)
    if has_weight:
        if weight is not None:
            weight = weight[None, :, None].expand(pred.size()).to(y_true.device)
        metric = partial(metric, weight=weight)  # type: ignore
    return metric(
        pred,
        torch.quantile(y_true, quantiles, dim=dim),
    )


def sinkhorn_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    backend: Literal["auto", "tensorized", "online", "multiscale"] = "auto",
    p: int = 2,
    blur: float = 0.1,
    scaling: float = 0.7,
    debias: bool = True,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the Sinkhorn divergence between two tensors.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted values.
    y_true : torch.Tensor
        True values.
    backend : Literal["auto", "tensorized", "online", "multiscale"], optional
        The implementation that will be used in the background.
        Default: "auto". Supported values:
        - "auto": Choose backend automatically.
        - "tensorized": Computes on a full cost/kernel matrix once and
            and stores it on the device memory.
        - "online": Computes cost/kernel values on-the-fly.
        - "multiscale": Fast implementation that scales to millions of samples.
    p : int, optional
        Specifies the ground cost function between points. Default: 2.
        Supported values:
        - p=1: Euclidean distance.
        - p=2: Squared Euclidean distance.
    blur : float, optional
        Scale associated with the temperature. Default: 0.1.
    scaling : float, optional
        Ratio of between successive values in the scaling descent. Default: 0.7.
    debias : bool, optional
        Compute unbiased Sinkhorn divergence. Default: True.
    weight : Optional[torch.Tensor], optional
        Weights for each sample. Default: None.

    Returns
    -------
    torch.Tensor
        Average Sinkhorn divergence across samples.
    """
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = SamplesLoss(
        loss="sinkhorn",
        backend=backend,
        p=p,
        blur=blur,
        scaling=scaling,
        debias=debias,
    )
    loss = loss_fn(y_pred, y_true)
    if weight is None:
        return loss.mean()
    else:
        weight = weight.to(y_true.device)
        return (loss @ weight) / weight.sum()


class LossFunction(Protocol):
    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor: ...


def integrate_model(
    y0: torch.Tensor,
    model: GRNModel,
    n_steps: Optional[int] = None,
    time_points: Optional[torch.Tensor] = None,
    adjoint: bool = False,
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    adjoint_atol: float = 1e-4,
    adjoint_rtol: float = 1e-5,
) -> torch.Tensor:
    """
    Integrate a gene regulatory differential equation model (ODE or SDE).

    Parameters
    ----------
    y0 : torch.Tensor
        Initial state of the differential equation of shape (batch_size, n_var).
    model : GRNModel
        The gene regulatory network model.
    n_steps : Optional[int], optional
        Number of steps to integrate. Mutually exclusive with
        `time_points`. Using this option will generate `n_steps` equally
        spaced time points from 0 to `n_steps` - 1. Default: None.
    time_points : Optional[torch.Tensor], optional
        Time points to use for integration. Mutually exclusive with
        `num_steps`. Allows explicitly specifying the time points.
        Default: None.
    adjoint : bool, optional
        Whether to use the adjoint method. Default: False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-5.

    Returns
    -------
    torch.Tensor
        State tensor of size (n_steps, batch_size, n_var).

    Raises
    ------
    TypeError
        An error is raised if the model is of the wrong type.
    """
    # Choose the correct integration method
    if isinstance(model, GRNModelSDE):
        if adjoint:
            int_func = partial(
                sdeint_adjoint,
                dt=dt,
                adjoint_atol=adjoint_atol,
                adjoint_rtol=adjoint_rtol,
            )
        else:
            int_func = partial(sdeint, dt=dt)
    elif isinstance(model, GRNModel):
        if adjoint:
            int_func = partial(
                odeint_adjoint, adjoint_atol=adjoint_atol, adjoint_rtol=adjoint_rtol
            )
        else:
            int_func = odeint
    else:
        raise TypeError(f"Expected type GRNModel, but got {type(model)}.")

    if n_steps is None:
        if time_points is None:
            raise ValueError(
                "Arguments `n_steps` and `time_points` are both `None`. You must set one of them."
            )
        else:
            ts = time_points.to(y0.device, dtype=y0.dtype)
    else:
        if time_points is not None:
            raise ValueError(
                "Arguments `n_steps` and `time_points` are mutually exclusive. Choose one."
            )
        else:
            ts = torch.linspace(
                0, n_steps - 1, n_steps, dtype=y0.dtype, device=y0.device
            )

    # Integrate
    pred = int_func(
        model,
        y0,
        ts,
        method=solver_method,
        atol=atol,
        rtol=rtol,
        options={"dtype": torch.float32},
    )
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)

    return pred


def train_step_integration(
    y: torch.Tensor,
    model: GRNModel,
    optimizer: torch.optim.Optimizer,
    loss_func: LossFunction,
    time_points: Optional[torch.Tensor] = None,
    adjoint: bool = False,
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    adjoint_atol: float = 1e-4,
    adjoint_rtol: float = 1e-5,
    loss_kwargs: dict = {},
) -> float:
    """
    A single training step for a gene regulatory network ODE/SDE model,
    including  integration, loss evaluation, backpropagation, and weight
    updates.

    Parameters
    ----------
    y : torch.Tensor
        Training sequences of shape (n_steps, batch_size, n_var).
    model : GRNModel
        The gene regulatory network model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer for model training.
    loss_func : LossFunction
        Loss function to optimize.
    time_points : Optional[torch.Tensor], optional
        Time points to use for integration. Allows explicitly specifying
        the time points. Default: None.
    adjoint : bool, optional
        Whether to use the adjoint method. Default: False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    float
        Value of the training loss after the training step.
    """
    n_steps = None
    if time_points is None:
        n_steps = y.shape[1]

    model.train()
    pred = integrate_model(
        y0=y[:, 0, :],
        model=model,
        n_steps=n_steps,
        time_points=time_points,
        adjoint=adjoint,
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
        adjoint_atol=adjoint_atol,
        adjoint_rtol=adjoint_rtol,
    )
    if torch.any(pred.isnan()):
        raise NaNTensorError("Numerical integration resulted in all NaN tensor.")

    # Backpropagation
    loss = loss_func(pred.contiguous(), y.permute(1, 0, 2).contiguous(), **loss_kwargs)
    if loss.dim() >= 1:
        loss = loss.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def test(
    y: torch.Tensor,
    model: GRNModel,
    loss_func: LossFunction,
    time_points: Optional[torch.Tensor] = None,
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    loss_kwargs: dict = {},
) -> float:
    """
    Evaluate a gene regulatory network ODE/SDE model on a set of test
    sequences by integration.

    Parameters
    ----------
    y : torch.Tensor
        Test sequences of shape (n_steps, n_obs, n_var).
    model : GRNModel
        The gene regulatory network model.
    loss_func : LossFunction
        Loss function to optimize.
    time_points : Optional[torch.Tensor], optional
        Time points to use for integration. Allows explicitly specifying
        the time points. Default: None.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    float
        Value of the test lost.
    """
    n_steps = None
    if time_points is None:
        n_steps = y.shape[1]

    model.eval()
    with torch.no_grad():
        pred = integrate_model(
            y0=y[:, 0, :],
            model=model,
            n_steps=n_steps,
            time_points=time_points,
            adjoint=False,
            solver_method=solver_method,
            dt=dt,
            atol=atol,
            rtol=rtol,
        )

        if torch.any(pred.isnan()):
            raise NaNTensorError("Numerical integration resulted in all NaN tensor.")

        loss = loss_func(
            pred.contiguous(), y.permute(1, 0, 2).contiguous(), **loss_kwargs
        )
        if loss.dim() >= 1:
            loss = loss.sum()
        return loss.item()


def train_integration(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    optimizer: torch.optim.Optimizer,
    loss_func: LossFunction,
    train_proportion: float = 0.8,
    batch_size: Optional[int] = None,
    num_epochs: int = 1000,
    early_stopping: bool = True,
    patience: int = 5,
    delta: float = 0.0,
    early_stopping_mode: Literal["min", "max"] = "min",
    use_accelerator: bool = True,
    adjoint: bool = False,
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    adjoint_atol: float = 1e-4,
    adjoint_rtol: float = 1e-5,
    loss_kwargs: dict = {},
    num_workers_loader: int = 0,
    checkpoint: Optional[Union[int, Literal["final"]]] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[float], List[float]]:
    """
    Train a gene regulatory network ODE/SDE model using integration.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for training.
    model : GRNModel
        The gene regulatory network model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer for model training.
    loss_func : LossFunction
        Loss function to optimize.
    train_proportion : float, optional
        Proportion of samples to use for training. Default: 0.8.
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
    early_stopping_mode : Literal["min", "max"], optional
        The direction of the optimization. In "min" mode, training will
        stop when the monitored quantity stops decreasing, in "max" mode
        when it stops increasing. Default: "min".
    use_accelerator : bool, optional
        Whether to use an accelerator (CUDA or MPS) if it is available.
        Default: True.
    adjoint : bool, optional
        Whether to use the adjoint method. Default: False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. Default: 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.
    num_workers_loader : int, optional
        Number of subprocesses to use for data loading. With the default,
        0, data is loaded in the main process.
    checkpoint : Optional[Union[int, Literal["final"]]], optional
        Save the model state every `checkpoint` epochs, or only save the
        state of the final model with "final". The default None
        corresponds to not saving any checkpoints.
    checkpoint_dir : Optional[Union[str, Path]], optional
        Directory for saving checkpoints. Default: None (will save
        checkpoints to the directory ./checkpoints if `checkpoint` is not
        None).
    logger : Optional[logging.Logger], optional
        Logger instance for logging output. Default: None (uses logger
        of this file).

    Returns
    -------
    Tuple[List[float], List[float]]
        Train and test losses for every epoch.

    Raises
    ------
    TypeError
        An error is raised if `checkpoint` is not of type int or if
        `checkpoint_dir` is not of type str or Path.
    ValueError
        An error is raised if `checkpoint` is larger than `num_epochs`.
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
            if checkpoint == "final":
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                log.info(f"Saving final model state to directory {checkpoint_dir}.")
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
        early_stopping_instance = EarlyStopping(
            patience, delta, early_stopping_mode, log
        )

    log.info(f"Using model of type {model.__class__.__name__}")

    # Select GPU if available
    device = None
    if use_accelerator and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    log.info(f"Running on device '{device}'")

    time_points = None
    if isinstance(dataset, PseudotimeDataset):
        time_points = dataset.time_points
    elif not isinstance(dataset, MarkovChainDataset):
        raise TypeError(
            "Incompatible dataset type. Use `MarkovChainDataset` or `PseudotimeDataset`."
        )

    train_length = floor(len(dataset) * train_proportion)
    trainset, testset = torch.utils.data.random_split(
        dataset, [train_length, len(dataset) - train_length]
    )

    # No batch size given: Use the whole dataset
    if batch_size is not None:
        batch_size_train = batch_size
        batch_size_test = batch_size
    else:
        batch_size_train = len(trainset)
        batch_size_test = len(testset)
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
        testset,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=num_workers_loader,
    )

    # Train for multiple epochs
    num_batches_train = len(trainloader)
    num_batches_test = len(testloader)
    train_losses: List[float] = []
    test_losses: List[float] = []
    log.info(
        f"Number of training batches: {num_batches_train}; number of test batches: {num_batches_test}"
    )

    log.info("Starting GRN model training.")

    with logging_redirect_tqdm():
        for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
            # Training
            train_loss = 0.0
            for data in trainloader:
                y = data.to(device)
                train_loss += train_step_integration(
                    y=y,
                    model=model,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    time_points=time_points,
                    adjoint=adjoint,
                    solver_method=solver_method,
                    dt=dt,
                    atol=atol,
                    rtol=rtol,
                    adjoint_atol=adjoint_atol,
                    adjoint_rtol=adjoint_rtol,
                    loss_kwargs=loss_kwargs,
                )
            train_loss = train_loss / num_batches_train
            train_losses.append(train_loss)

            # Evaluation
            test_loss = 0.0
            for data in testloader:
                y = data.to(device)
                test_loss += test(
                    y=y,
                    model=model,
                    loss_func=loss_func,
                    time_points=time_points,
                    solver_method=solver_method,
                    dt=dt,
                    atol=atol,
                    rtol=rtol,
                    loss_kwargs=loss_kwargs,
                )
            test_loss = test_loss / num_batches_test
            test_losses.append(test_loss)

            log.debug(
                f"Epoch {epoch}: training loss: {train_loss}, test loss: {test_loss}"
            )

            # Save checkpoint
            if isinstance(checkpoint, int) and epoch % checkpoint == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    },
                    checkpoint_dir
                    / f"checkpoint_e{epoch:0{floor(log10(num_epochs)) + 1}d}.tar",
                )

            # Check early stopping
            if early_stopping:
                if early_stopping_instance.should_stop(test_loss):
                    break

        log.info("Finished model training.")

        # Save final model
        if checkpoint is not None:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                checkpoint_dir / "checkpoint_final_model.tar",
            )

        return train_losses, test_losses
