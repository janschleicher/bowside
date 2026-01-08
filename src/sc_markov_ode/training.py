"""Module for training of dynamic GRN model."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple, Union

import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from sc_markov_ode.dataset import MarkovChainDataset
from sc_markov_ode.model import GRNModel, GRNModelSDE

log = logging.getLogger(__name__)


def qqloss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    dim: int = 0,
    n_quantiles: int = 100,
    metric: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = torch.nn.functional.mse_loss,
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

    Returns
    -------
    torch.Tensor
        Difference between quantiles.
    """
    quantiles = torch.linspace(0, 1, n_quantiles + 1, device=y_true.device)
    return metric(
        torch.quantile(y_pred, quantiles, dim=dim),
        torch.quantile(y_true, quantiles, dim=dim),
    )


class LossFunction(Protocol):
    def __call__(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor: ...


def integrate_model(
    y0: torch.Tensor,
    n_steps: int,
    model: GRNModel,
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
    n_steps : int
        Number of steps to integrate.
    model : GRNModel
        The gene regulatory network model.
    adjoint : bool, optional
        Whether to use the adjoint method. By default False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. By default 0.001.
    atol : float, optional
        Absolute tolerance. By default 1e-4.
    rtol : float, optional
        Relative tolerance. By default 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-5.

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

    # Integrate and compute loss
    pred = int_func(
        model,
        y0,
        torch.linspace(0, n_steps - 1, n_steps, dtype=torch.float32, device=y0.device),
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
    adjoint : bool, optional
        Whether to use the adjoint method. By default False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. By default 0.001.
    atol : float, optional
        Absolute tolerance. By default 1e-4.
    rtol : float, optional
        Relative tolerance. By default 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. By default {}.

    Returns
    -------
    float
        Value of the training loss after the training step.
    """
    model.train()
    pred = integrate_model(
        y0=y[0, :, :],
        n_steps=y.shape[0],
        model=model,
        adjoint=adjoint,
        solver_method=solver_method,
        dt=dt,
        atol=atol,
        rtol=rtol,
        adjoint_atol=adjoint_atol,
        adjoint_rtol=adjoint_rtol,
    )

    # Backpropagation
    loss = loss_func(pred, y, **loss_kwargs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def test(
    y: torch.Tensor,
    model: GRNModel,
    loss_func: LossFunction,
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
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. By default 0.001.
    atol : float, optional
        Absolute tolerance. By default 1e-4.
    rtol : float, optional
        Relative tolerance. By default 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. By default {}.

    Returns
    -------
    float
        Value of the test lost.
    """
    model.eval()
    with torch.no_grad():
        pred = integrate_model(
            y0=y[0, :, :],
            n_steps=y.shape[0],
            model=model,
            adjoint=False,
            solver_method=solver_method,
            dt=dt,
            atol=atol,
            rtol=rtol,
        )
        return loss_func(pred, y, **loss_kwargs).item()


def train_integration(
    dataset: MarkovChainDataset,
    model: GRNModel,
    optimizer: torch.optim.Optimizer,
    loss_func: LossFunction,
    train_proportion: float = 0.8,
    batch_size: Optional[int] = None,
    num_epochs: int = 1000,
    adjoint: bool = False,
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
    adjoint_atol: float = 1e-4,
    adjoint_rtol: float = 1e-5,
    loss_kwargs: dict = {},
    num_workers_loader: int = 0,
    checkpoint: Optional[int] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Train a gene regulatory network ODE/SDE model using integration.

    Parameters
    ----------
    dataset : MarkovChainDataset
        Dataset to use for training.
    model : GRNModel
        The gene regulatory network model.
    optimizer : torch.optim.Optimizer
        PyTorch optimizer for model training.
    loss_func : LossFunction
        Loss function to optimize.
    train_proportion : float, optional
        Proportion of samples to use for training. By default 0.8.
    batch_size : Optional[int], optional
        Batch size. With the default None, all samples are used at once.
    num_epochs : int, optional
        Number of training epochs. By default 1000.
    adjoint : bool, optional
        Whether to use the adjoint method. By default False.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the mode type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. By default 0.001.
    atol : float, optional
        Absolute tolerance. By default 1e-4.
    rtol : float, optional
        Relative tolerance. By default 1e-5.
    adjoint_atol : float, optional
        Absolute tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-4.
    adjoint_rtol : float, optional
        Relative tolerance for backward adjoint solve. Only used when
        `adjoint=True`. By default 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. By default {}.
    num_workers_loader : int, optional
        Number of subprocesses to use for data loading. With the default,
        0, data is loaded in the main process.
    checkpoint : int, optional
        Save the model state every `checkpoint` epochs. The default None
        corresponds to not saving any checkpoints.

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
    if checkpoint is not None:
        if not isinstance(checkpoint, int):
            raise TypeError(
                f"Argument `checkpoint` must be of type int, but got {type(checkpoint)}."
            )
        elif checkpoint > num_epochs:
            raise ValueError(
                f"Argument `checkpoint` must be smaller than `num_epochs`. "
                f"Got checkpoint={checkpoint}, num_epochs={num_epochs}."
            )

    if checkpoint_dir is None:
        checkpoint_dir = Path("./checkpoints")
    elif isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    elif not isinstance(checkpoint_dir, Path):
        raise TypeError(
            f"Argument `checkpoint_dir` must be of type str or Path, "
            f"but got {type(checkpoint_dir)}"
        )

    # Select GPU if available
    device = torch.accelerator.current_accelerator()
    if not device:
        device = torch.device("cpu")
    model = model.to(device)

    trainset, testset = torch.utils.data.random_split(
        dataset, [train_proportion, 1.0 - train_proportion]
    )

    # No batch size given: Use the whole dataset
    if batch_size:
        batch_size_train = batch_size
        batch_size_test = batch_size
    else:
        batch_size_train = len(trainset)
        batch_size_test = len(testset)

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

    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        for data in trainloader:
            y = data.to(device)
            train_loss += train_step_integration(
                y=y,
                model=model,
                optimizer=optimizer,
                loss_func=loss_func,
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
                solver_method=solver_method,
                dt=dt,
                atol=atol,
                rtol=rtol,
                loss_kwargs=loss_kwargs,
            )
        test_loss = test_loss / num_batches_test
        test_losses.append(test_loss)

        log.info(f"Epoch {epoch}: training loss: {train_loss}, test loss: {test_loss}")

        # Save checkpoint
        if checkpoint is not None and epoch % checkpoint == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                },
                checkpoint_dir / f"checkpoint_e{epoch:4d}.tar",
            )

    return train_losses, test_losses
