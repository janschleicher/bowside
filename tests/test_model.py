from typing import Literal, Optional

import pytest
import torch

from bowside.model import GRNModel, GRNModelSDE

TEST_ADJ_MAT = torch.tensor(
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
    adj: Optional[torch.Tensor],
    adj_init_normal: bool,
    adj_sparsity: float,
    activation: Literal["sigmoid", "identity", "softplus", "relu"],
):
    assert model.adj.size() == (n_genes, n_genes), (
        "Mismatch between number of genes and adjacency matrix size."
    )
    if adj is not None:
        assert torch.all((model.adj != 0) == (adj != 0)), (
            "The adjacency matrix in the untrained model should only be nonzero where "
            "the prior is nonzero."
        )
        assert torch.all(model.adj_mask == (adj != 0).int()), (
            "The adjacency matrix mask should be one where the mask is nonzero."
        )
        if not adj_init_normal:
            assert torch.all(model.adj == adj), (
                "Without intitialization from a normal distribution, the adjacency "
                "matrix should be the same as the prior."
            )
    elif n_genes >= 100:  # Only check sparsity for test cases with many genes
        adj_sparsity_model = (model.adj == 0).sum() / (n_genes**2)
        assert torch.allclose(adj_sparsity_model, torch.tensor(adj_sparsity)), (
            f"The adjacency matrix has the wrong sparsity; should be {adj_sparsity}, but is {adj_sparsity_model}."
        )
    assert model.activation.__class__.__name__.lower() == activation, (
        f"Wrong activation function, should be {activation}, but is {model.activation.__class__.__name__.lower()}."
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
        # TODO: add tests for sparse init (w/o adj!)
    ],
)
class TestGRNModels:
    def test_GRNModel(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor],
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
        )
        check_model(
            model=model,
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )

    def test_GRNModelSDE(
        self,
        n_genes: int,
        adj: Optional[torch.Tensor],
        adj_init_normal: bool,
        adj_sparsity: float,
        activation: Literal["sigmoid", "identity", "softplus", "relu"],
    ):
        model = GRNModelSDE(
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )
        check_model(
            model=model,
            n_genes=n_genes,
            adj=adj,
            adj_init_normal=adj_init_normal,
            adj_sparsity=adj_sparsity,
            activation=activation,
        )
