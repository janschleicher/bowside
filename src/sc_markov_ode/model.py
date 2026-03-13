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
        normal distribution. Default: False (initializes with ones).
    adj_sparsity : float, optional
        Sparsity level of initial, random adjacency matrix if `adj` is
        not provided. Default: 0.9.
    activation : Literal["sigmoid", "identity", "softplus", "relu"], optional
        Activation function for the production rates. Default: "sigmoid".
    """

    def __init__(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor] = None,
        adj_init_normal: bool = False,
        adj_sparsity: float = 0.9,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
    ):
        super().__init__()

        # Initialize adjacency matrix; adj[i, j] is the edge from gene i to gene j
        self.adj = torch.ones(n_genes, n_genes)
        if adj_init_normal:
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

            # Ensure that adj has same signs as adj_init
            self.adj = self.adj.abs() * adj

        else:
            sparse_mask = torch.empty_like(self.adj)
            torch.nn.init.sparse_(
                sparse_mask, sparsity=adj_sparsity, std=1
            )  # TODO: find right sparsity level
            self.adj = self.adj * (sparse_mask != 0)
            self.adj_mask = torch.ones_like(self.adj)
        self.adj = torch.nn.Parameter(self.adj)
        self.adj_mask = torch.nn.Parameter(self.adj_mask, requires_grad=False)

        # Use log of bias and decay rates to ensure positivity
        # TODO: consider using parametrization instead
        # (https://docs.pytorch.org/tutorials/intermediate/parametrizations.html)
        self.log_bias = torch.empty(n_genes)
        torch.nn.init.normal_(self.log_bias, mean=0, std=0.01)
        self.log_bias = torch.nn.Parameter(self.log_bias)
        self.log_decay_rate = torch.empty(n_genes)
        torch.nn.init.normal_(self.log_decay_rate, mean=0, std=0.01)
        self.log_decay_rate = torch.nn.Parameter(self.log_decay_rate)
        self.log_scale = torch.nn.Parameter(torch.zeros(n_genes))

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
            self.log_scale.exp()
            * self.activation(
                ((self.adj * self.adj_mask).T @ x.T).T + self.log_bias.exp()
            )
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
    adj_sparsity : float, optional
        Sparsity level of initial, random adjacency matrix if `adj` is
        not provided. Default: 0.9.
    activation : Literal["sigmoid", "identity", "softplus", "relu"], optional
        Activation function for the production rates. Default: "sigmoid".
    sde_type : Literal["ito", "stratonovich"], optional
        Type of the SDE, depending on the underlying type of integral.
        Default: "ito".
    sigma : float, optional
        Level of the (diagonal) noise. Default: 0.1.
    shared_sigma : bool, opional
        Whether to share sigma across all genes. Default: True.
    fixed_sigma : bool, optional
        Whether to use a fixed value of sigma. If set to False, sigma
        will be a learnable parameter. Only implemented for
        `shared_sigma=True`. Default: False.
    constant_noise : bool, optional
        Whether the noise should be constant (sigma). The alternative is
        to multiply sigma with the expression level. Default: True.
    """

    def __init__(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor] = None,
        adj_init_normal: bool = False,
        adj_sparsity: float = 0.9,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
        sde_type: Literal["ito", "stratonovich"] = "ito",
        sigma: float = 0.1,
        shared_sigma: bool = True,
        fixed_sigma: bool = False,
        constant_noise: bool = True,
    ):
        super().__init__(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )
        self.noise_type = "diagonal"
        self.sde_type = sde_type
        if shared_sigma:
            self.log_sigma = torch.nn.Parameter(
                torch.log(torch.tensor([sigma])), requires_grad=(not fixed_sigma)
            )
        else:
            if fixed_sigma:
                raise NotImplementedError("Fixed, gene-wise sigma is not implemented.")
            else:
                self.log_sigma = torch.nn.Parameter(
                    torch.log(torch.ones((n_genes)) * sigma)
                )
        self.constant_noise = torch.nn.Buffer(torch.tensor(constant_noise))

    def f(self, t, x):
        return super().forward(t, x)

    def g(self, t, x):
        sigma = self.log_sigma.exp().expand(x.size(0), x.size(1))
        if self.constant_noise:
            return sigma
        else:
            return sigma * x
