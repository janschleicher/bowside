"""Module for training of dynamic GRN model."""

from typing import Callable

import torch


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
    metric: Callable = mean_absolute_error,
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
