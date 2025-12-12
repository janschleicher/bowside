"""Module for training of dynamic GRN model."""

from collections.abc import Callable
from functools import partial
from typing import Any, Optional, Protocol

import torch
from torchdiffeq import odeint, odeint_adjoint
from torchsde import sdeint, sdeint_adjoint

from sc_markov_ode.dataset import MarkovChainDataset
from sc_markov_ode.model import GRNModel, GRNModelSDE


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between two tensors.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Mean squared error.
    """
    return torch.mean((y_pred - y_true) ** 2)


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error between two tensors.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Mean absolute error.
    """
    return torch.mean((y_pred - y_true).abs())


def qqloss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int = 0,
    n_quantiles: int = 100,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mean_absolute_error,
) -> torch.Tensor:
    """
    Compute the difference between the quantiles of two tensors along a
    given dimension.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    dim : int, optional
        The dimension to reduce. Default: 0.
    n_quantiles : int, optional
        Number of quantiles. Default: 100.
    metric : Callable, optional
        Metric for comparing the quantiles. Default: `mean_absolute_error`.

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
        self, y_true: torch.Tensor, y_pred: torch.Tensor, *args: Any, **kwargs: Any
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
        torch.linspace(0, n_steps - 1, n_steps, dtype=torch.float32),
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
    loss = loss_func(y, pred, **loss_kwargs)
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
        return loss_func(y, pred, loss_kwargs).item()


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
) -> None:
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
    """
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
    for epoch in range(num_epochs):
        # Training
        running_loss = 0.0
        for data in trainloader:
            y = data.to(device)
            running_loss += train_step_integration(
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

        print(
            f"Epoch {epoch}: training loss: {running_loss / num_batches_train}, test loss: {test_loss / num_batches_test}"
        )
