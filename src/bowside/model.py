"""Module for dynamic gene regulatory network model."""

import json
from pathlib import Path
from typing import Callable, Literal, Optional, Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import diffrax
import equinox as eqx
import jax
import lineax as lx
from jax import numpy as jnp


class GRNModel(eqx.Module):
    """
    Gene regulatory network ordinary differential equation model.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    adj : Optional[jax.Array], optional
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
    solver : diffrax.AbstractSolver, optional
        Differential equation solver. Default: diffrax.Tsit5().
    solver_atol : float, optional
        Absolute error tolerance for adaptive solvers. Default: 1e-4.
    solver_rtol : float, optional
        Relative error tolerance for adaptive solvers. Default: 1e-5.
    dt : float, optional
        The step size for the solver. If an adaptive solver is used,
        this is the initial step size. Default: 0.01
    key : jax.Array
        Key for the random number generator.
    """

    adj: jax.Array
    adj_mask: jax.Array
    adj_init: jax.Array
    log_bias: jax.Array
    log_decay_rate: jax.Array
    log_scale: jax.Array
    activation: Callable[[jax.Array], jax.Array]
    solver: diffrax.AbstractSolver
    dt: float
    step_size_controller: diffrax.AbstractStepSizeController
    hyperparams: dict

    def __init__(
        self,
        n_genes: int,
        adj: Optional[jax.Array] = None,
        adj_init_normal: bool = False,
        adj_sparsity: float = 0.9,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
        solver: diffrax.AbstractSolver = diffrax.Tsit5(),
        solver_atol: float = 1e-4,
        solver_rtol: float = 1e-5,
        dt: float = 0.01,
        *,
        key: jax.Array,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Hyperparameters to facilitate model saving and loading
        self.hyperparams = {
            "n_genes": n_genes,
            "activation": activation,
            "solver_name": solver.__class__.__name__,
            "solver_atol": solver_atol,
            "solver_rtol": solver_rtol,
            "dt": dt,
        }

        adj_key, sparse_key, bias_key, decay_key = jax.random.split(key, 4)

        # Initialize adjacency matrix; adj[i, j] is the edge from gene i to gene j
        self.adj = jnp.ones((n_genes, n_genes))
        if adj_init_normal:
            initializer = jax.nn.initializers.normal(stddev=1.0)
            self.adj = initializer(adj_key, self.adj.shape)
        if adj is not None:
            assert adj.shape == (n_genes, n_genes), (
                "Adjacency matrix must be of shape (n_genes, n_genes)."
            )
            assert jnp.isin(adj, jnp.asarray([-1, 0, 1])).all(), (
                "Adjacency matrix for initialization may only contain values -1, 0, and 1."
            )

            self.adj_init = adj.copy()

            # Mask to restrict it to only those edges that are in the prior
            self.adj_mask = adj != 0

            # Ensure that adj has same signs as adj_init
            self.adj = jnp.abs(self.adj) * adj

        else:
            sparse_mask = (
                jax.random.uniform(sparse_key, self.adj.shape, minval=0.0, maxval=1.0)
                > adj_sparsity
            ).astype(int)
            self.adj = self.adj * sparse_mask
            self.adj_init = self.adj.copy()
            self.adj_mask = jnp.ones_like(self.adj, dtype=jnp.bool)

        # Use log of bias, decay rates, and scale to ensure positivity
        initializer = jax.nn.initializers.normal(stddev=0.01)
        self.log_bias = initializer(bias_key, (n_genes,))
        self.log_decay_rate = initializer(decay_key, (n_genes,))
        self.log_scale = jnp.zeros(n_genes)

        match activation:
            case "sigmoid":
                self.activation = jax.nn.sigmoid
            case "identity":
                self.activation = jax.nn.identity
            case "softplus":
                self.activation = jax.nn.softplus
            case "relu":
                self.activation = jax.nn.relu

        self.solver = solver
        self.dt = dt
        if isinstance(solver, diffrax.AbstractAdaptiveSolver):
            self.step_size_controller = diffrax.PIDController(
                rtol=solver_rtol, atol=solver_atol
            )
        else:
            self.step_size_controller = diffrax.ConstantStepSize()

    def drift(self, t, y, args):
        return (
            jnp.exp(self.log_scale)
            * self.activation(
                y @ (self.adj * jax.lax.stop_gradient(self.adj_mask))
                + jnp.exp(self.log_bias)
            )
            - jnp.exp(self.log_decay_rate) * y
        )

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.drift),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt,
            y0=y0,
            stepsize_controller=self.step_size_controller,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

    def prune(self, atol: float = 1e-3, rtol: float = 1e-3) -> Self:
        """
        Prune edges with a weight close to 0.

        Parameters
        ----------
        atol : float, optional
            Absolute threshold for pruning edges close to 0. Default: 1e-3.
        rtol : float, optional
            Relative threshold for pruning edges close to 0. Default: 1e-3.

        Returns
        -------
        GRNModel
            Pruned model.
        """
        adj_mask = self.adj_mask * ~jnp.isclose(self.adj, 0, rtol=rtol, atol=atol)
        model = eqx.tree_at(lambda m: m.adj_mask, self, adj_mask)
        adj = model.adj * model.adj_mask
        return eqx.tree_at(lambda m: m.adj, model, adj)

    def n_genes(self) -> int:
        return self.adj.shape[0]

    def save(self, filename: Union[str, Path]):
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(self.hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, filename: Union[str, Path]) -> Self:
        with open(filename, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            solver_name = hyperparams.pop("solver_name")
            model = cls(
                **hyperparams,
                solver=getattr(diffrax, solver_name)(),
                key=jax.random.key(0),
            )
            return eqx.tree_deserialise_leaves(f, model)


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
    solver : diffrax.AbstractSolver, optional
        Differential equation solver. Default: diffrax.ItoMilstein().
    solver_atol : float, optional
        Absolute error tolerance for adaptive solvers. Default: 1e-4.
    solver_rtol : float, optional
        Relative error tolerance for adaptive solvers. Default: 1e-5.
    dt : float, optional
        The step size for the solver. If an adaptive solver is used,
        this is the initial step size. Default: 0.01
    key : jax.Array
        Key for the random number generator.
    """

    fixed_sigma: bool
    log_sigma: jax.Array
    constant_noise: bool

    def __init__(
        self,
        n_genes: int,
        adj: Optional[jax.Array] = None,
        adj_init_normal: bool = False,
        adj_sparsity: float = 0.9,
        activation: Literal["sigmoid", "identity", "softplus", "relu"] = "sigmoid",
        sigma: float = 0.1,
        shared_sigma: bool = True,
        fixed_sigma: bool = False,
        constant_noise: bool = True,
        solver: diffrax.AbstractSolver = diffrax.ItoMilstein(),
        solver_atol: float = 1e-4,
        solver_rtol: float = 1e-5,
        dt: float = 0.01,
        *,
        key: jax.Array,
        **kwargs,
    ):
        super().__init__(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
            solver=solver,
            solver_atol=solver_atol,
            solver_rtol=solver_rtol,
            dt=dt,
            key=key,
            **kwargs,
        )
        self.hyperparams.update(
            {
                "sigma": sigma,
                "shared_sigma": shared_sigma,
                "fixed_sigma": fixed_sigma,
                "constant_noise": constant_noise,
            }
        )
        self.fixed_sigma = fixed_sigma
        self.log_sigma = jnp.log(sigma)
        if not shared_sigma:
            if fixed_sigma:
                raise NotImplementedError("Fixed, gene-wise sigma is not implemented.")
            else:
                self.log_sigma = jnp.broadcast_to(self.log_sigma, n_genes)
        self.constant_noise = constant_noise

    def diffusion(self, t, y, args):
        if self.fixed_sigma:
            sigma = jnp.broadcast_to(
                jnp.exp(jax.lax.stop_gradient(self.log_sigma)), y.shape
            )
        else:
            sigma = jnp.broadcast_to(jnp.exp(self.log_sigma), y.shape)
        if self.constant_noise:
            diag = sigma
        else:
            diag = sigma * y
        return lx.DiagonalLinearOperator(diag)

    def __call__(self, ts, y0, key):
        brownian = diffrax.VirtualBrownianTree(
            ts[0], ts[-1], tol=self.dt / 2, shape=y0.shape, key=key
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(self.drift), diffrax.ControlTerm(self.diffusion, brownian)
        )
        solution = diffrax.diffeqsolve(
            terms,
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=self.dt,
            y0=y0,
            stepsize_controller=self.step_size_controller,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


class BifurcatingSDE(GRNModelSDE):
    """
    Dummy SDE generating a bifurcating system with two genes.
    """

    def __init__(self):
        super().__init__(2, dt=0.1, key=jax.random.key(42))
        self.adj = jnp.asarray([[2, -2.0], [-2.0, 2.0]])
        self.log_bias = jnp.log(jnp.asarray([1.0, 1.0]))
        self.log_decay_rate = jnp.log(jnp.asarray([0.6, 0.6]))
