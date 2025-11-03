"""Module for dynamic gene regulatory network model."""

from typing import Literal, Optional

import torch


class GRNModel(torch.nn.Module):
    """
    Gene regulatory network ordinary differential equation model.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    adj : Optional[torch.Tensor], optional
        Adjacency matrix initialization. Only the values -1 (inhibition),
        0 (no regulation), and 1 (activation) are allowed. The value of
        adj[i, j] determines if there is an edge from gene i to gene j.
        Default: None.
    adj_init_normal : bool, optional
        Whether to initialize the weighted adjacency matrix from a
        normal distribution if `adj` is provided. Default: False.
    activation : Literal["sigmoid", "identity", "softplus", "relu"], optional
        Activation function for the production rates. Default: "sigmoid".
    """

    def __init__(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor] = None,
        adj_init_normal: bool = False,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
    ):
        super().__init__()

        # Initialize adjacency matrix; adj[i, j] is the edge from gene i to gene j
        self.adj = torch.empty(n_genes, n_genes)
        torch.nn.init.normal_(self.adj, mean=0, std=1)
        if adj is not None:
            self.adj_init = torch.empty(n_genes, n_genes).copy_(adj)
            self.adj_init = torch.nn.Parameter(self.adj_init, requires_grad=False)
            assert adj.shape == (n_genes, n_genes), (
                "Adjacency matrix must be of shape (n_genes, n_genes)."
            )
            assert torch.isin(adj, torch.tensor([-1, 0, 1])).all(), (
                "Adjacency matrix for initialization may only contain values -1, 0, and 1."
            )

            # Mask to restrict it to only those edges that are in the prior
            self.adj_mask = (adj != 0).int()

            if adj_init_normal:
                self.adj = self.adj.abs() * adj
            else:
                self.adj.copy_(adj)
        else:
            sparse_mask = torch.empty_like(self.adj)
            torch.nn.init.sparse_(
                sparse_mask, sparsity=0.9
            )  # TODO: find right sparsity level
            self.adj = self.adj * (sparse_mask > 0)
            self.adj_mask = torch.ones_like(self.adj)  # TODO: remove
        self.adj = torch.nn.Parameter(self.adj)
        self.adj_mask = torch.nn.Parameter(self.adj_mask, requires_grad=False)
        self.bias = torch.empty(n_genes)
        torch.nn.init.normal_(self.bias, mean=0, std=0.01)
        self.bias = torch.nn.Parameter(self.bias)

        # Use log of decay rates to ensure positivity
        # TODO: consider using parametrization instead
        # (https://docs.pytorch.org/tutorials/intermediate/parametrizations.html)
        self.log_decay_rate = torch.empty(n_genes)
        torch.nn.init.normal_(self.log_decay_rate, mean=0, std=1)
        self.log_decay_rate = torch.nn.Parameter(self.log_decay_rate)

        match activation:
            case "sigmoid":
                self.activation = torch.nn.Sigmoid()
            case "identity":
                self.activation = torch.nn.Identity()
            case "softplus":
                self.activation = torch.nn.Softplus()
            case "relu":
                self.activation = torch.nn.ReLU()

    def forward(self, t, x):
        # TODO: Add scaling factor inside sigmoid function so it does not saturate so easily
        # or initialize adj with smaller weights to facilitate training
        return (
            self.activation(((self.adj * self.adj_mask).T @ x.T).T + self.bias)
            - self.log_decay_rate.exp() * x
        )


class GRNModelSDE(GRNModel):
    """
    Gene regulatory network stochastic differential equation model.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    adj : Optional[torch.Tensor], optional
        Adjacency matrix initialization. Only the values -1 (inhibition),
        0 (no regulation), and 1 (activation) are allowed. The value of
        adj[i, j] determines if there is an edge from gene i to gene j.
        Default: None.
    adj_init_normal : bool, optional
        Whether to initialize the weighted adjacency matrix from a
        normal distribution if `adj` is provided. Default: False.
    activation : Literal["sigmoid", "identity", "softplus", "relu"], optional
        Activation function for the production rates. Default: "sigmoid".
    sde_type: Literal["ito", "stratonovich"], optional
        Type of the SDE, depending on the underlying type of integral.
        Default: "ito".
    sigma: float, optional
        Level of the (diagonal) noise. Default: 0.1.
    """

    def __init__(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor] = None,
        adj_init_normal: bool = False,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
        sde_type: Literal["ito", "stratonovich"] = "ito",
        sigma: float = 0.1,
    ):
        super().__init__(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            activation=activation,
        )
        self.noise_type = "diagonal"
        self.sde_type = sde_type
        self.register_buffer("sigma", torch.tensor([[sigma]]))

    def f(self, t, x):
        return super().forward(t, x)

    def g(self, t, x):
        return self.sigma.repeat(x.size(0), x.size(1))
