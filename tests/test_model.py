from typing import Literal, Optional

import jax
import pytest
from jax import numpy as jnp

from bowside.model import GRNModel, GRNModelSDE

TEST_ADJ_MAT = jnp.asarray(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, -1],
        [0, 0, 0, -1, 0],
    ]
)


def check_model(
    model: GRNModel,
    n_genes: int,
    adj: Optional[jax.Array],
    adj_init_normal: bool,
    adj_sparsity: float,
    activation: Literal["sigmoid", "identity", "softplus", "relu"],
):
    assert model.adj.shape == (n_genes, n_genes), (
        "Mismatch between number of genes and adjacency matrix size."
    )
    if adj is not None:
        assert jnp.all((model.adj != 0) == (adj != 0)), (
            "The adjacency matrix in the untrained model should only be nonzero where "
            "the prior is nonzero."
        )
        assert jnp.all(model.adj_mask == (adj != 0).astype(int)), (
            "The adjacency matrix mask should be one where the mask is nonzero."
        )
        if not adj_init_normal:
            assert jnp.all(model.adj == adj), (
                "Without intitialization from a normal distribution, the adjacency "
                "matrix should be the same as the prior."
            )
    elif n_genes >= 100:  # Only check sparsity for test cases with many genes
        adj_sparsity_model = (model.adj == 0).sum() / (n_genes**2)
        assert jnp.isclose(adj_sparsity_model, adj_sparsity, atol=1e-3, rtol=1e-3), (
            f"The adjacency matrix has the wrong sparsity; should be {adj_sparsity}, but is {adj_sparsity_model}."
        )
    assert model.activation.__name__.lower() == activation, (
        f"Wrong activation function, should be {activation}, but is {model.activation.__name__.lower()}."
    )


@pytest.mark.parametrize(
    "n_genes, adj, adj_init_normal, adj_sparsity, activation",
    [
        [5, None, True, 0.9, "sigmoid"],
        [5, TEST_ADJ_MAT, True, 0.9, "identity"],
        [5, TEST_ADJ_MAT, False, 0.9, "softplus"],
        [5, TEST_ADJ_MAT, True, 0.9, "relu"],
        [1_000, None, False, 0.9, "sigmoid"],
        [1_000, None, False, 0.65, "sigmoid"],
    ],
)
class TestGRNModels:
    def test_GRNModel(
        self,
        n_genes: int,
        adj: Optional[jax.Array],
        adj_init_normal: bool,
        adj_sparsity: float,
        activation: Literal["sigmoid", "identity", "softplus", "relu"],
    ):
        model = GRNModel(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
            key=jax.random.key(42),
        )
        check_model(
            model=model,
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )

        # Test numerical integration
        ts = jnp.linspace(0, 1, 11)
        y0 = jnp.zeros(n_genes)
        solutions = model(ts, y0)
        assert isinstance(solutions, jax.Array)
        assert solutions.shape == (ts.shape[0], n_genes), (
            "Wrong shape of numerical integration solution."
        )

    def test_GRNModelSDE(
        self,
        n_genes: int,
        adj: Optional[jax.Array],
        adj_init_normal: bool,
        adj_sparsity: float,
        activation: Literal["sigmoid", "identity", "softplus", "relu"],
    ):
        key = jax.random.key(42)
        model_key, brownian_key = jax.random.split(key)
        model = GRNModelSDE(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
            key=model_key,
        )
        check_model(
            model=model,
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )

        # Test numerical integration
        ts = jnp.linspace(0, 1, 11)
        y0 = jnp.zeros(n_genes)
        solutions = model(ts, y0, brownian_key)
        assert isinstance(solutions, jax.Array)
        assert solutions.shape == (ts.shape[0], n_genes), (
            "Wrong shape of numerical integration solution."
        )


def test_forward():
    """
    Sanity check: New model formulation should produce same result as
    the old one (with double matrix transpose).
    """
    model = GRNModel(n_genes=5, adj=TEST_ADJ_MAT, key=jax.random.key(42))
    key = jax.random.key(7)
    x = jax.random.uniform(key, (20, model.n_genes()))
    dx = model.drift(None, x, None)

    dx_manual = (
        jnp.exp(model.log_scale)
        * model.activation(
            ((model.adj * model.adj_mask).T @ x.T).T + jnp.exp(model.log_bias)
        )
        - jnp.exp(model.log_decay_rate) * x
    )
    assert jnp.all(dx == dx_manual)
