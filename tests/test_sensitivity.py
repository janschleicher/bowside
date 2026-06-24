from typing import Literal, Union

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from bowside.dataset import BifurcatingMarkovChainDataset, BifurcatingPseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.sensitivity import simulate_all_knockouts, simulate_knockout
from bowside.training import qqloss
from bowside.trajectory import infer_trajectories

Y0 = jnp.asarray([[0.0, 0.0]]).repeat(500, 0)


@pytest.mark.parametrize("dataset_type", ["pseudotime", "markov"])
@pytest.mark.parametrize(
    "gene, simulate_before",
    [(0, False), (1, False), ("gene_0", False), ("gene_1", True)],
)
def test_simulate_knockout(
    dataset_type: Literal["pseudotime", "markov"],
    gene: Union[str, int],
    simulate_before: bool,
):
    if dataset_type == "pseudotime":
        dataset = BifurcatingPseudotimeDataset(Y0)
    elif dataset_type == "markov":
        dataset = BifurcatingMarkovChainDataset(Y0)
    model = dataset.sde

    # Sanity check
    assert "bowside" not in dataset.adata.uns

    key = jax.random.key(42)

    if simulate_before:
        key, sub_key = jax.random.split(key)
        infer_trajectories(dataset, model, key=sub_key, min_clusters=1, max_clusters=1)

    sequences_wt, sequences_ko, diff = simulate_knockout(
        dataset, model, gene, qqloss, key=key, loss_kwargs={"dim": 0}
    )
    adata = dataset.adata
    assert "bowside" in adata.uns, "Field 'bowside' missing from `adata.uns`."

    results = adata.uns["bowside"]
    assert "simulations" in results, (
        "Wild type simulations are missing from the results."
    )
    assert np.all(np.asarray(sequences_wt) == results["simulations"]), (
        "Mismatch between returned sequences and those saved to adata for wild type."
    )

    ko_simulations_key = "simulations_ko_" + (
        f"gene_{gene}" if isinstance(gene, int) else gene
    )
    assert ko_simulations_key in results, (
        "Knockout simulations are missing from the results."
    )
    assert np.all(np.asarray(sequences_ko) == results[ko_simulations_key]), (
        "Mismatch between returned sequences and those saved to adata for knockout."
    )

    gene_idx = gene if isinstance(gene, int) else dataset.gene_names.get_loc(gene)
    assert isinstance(gene_idx, int)
    assert jnp.all(sequences_ko[:, :, gene_idx] == 0), (
        "The expression of the knockout gene should be 0."
    )

    assert sequences_wt.shape == sequences_ko.shape, (
        "Wild type and knockout simulations are not of the same shape."
    )

    assert diff > 0, "Knockout effect should be greater than 0."


def test_simulate_knockout_exceptions():
    dataset = BifurcatingPseudotimeDataset(Y0)
    with pytest.raises(TypeError, match="Invalid dataset type"):
        simulate_knockout(dataset.adata, dataset.sde, 0, qqloss)  # type: ignore

    key_init, key_sim = jax.random.split(jax.random.key(42))
    model = GRNModelSDE(3, key=key_init)
    with pytest.raises(ValueError, match="Incompatible number of genes"):
        simulate_knockout(dataset, model, 0, qqloss, key_sim)

    model = dataset.sde
    with pytest.raises(KeyError, match="Gene .+ not found in the dataset"):
        simulate_knockout(dataset, model, "Nfkb", qqloss, key_sim)
    with pytest.raises(IndexError, match="Gene index 3 out of bounds for dataset"):
        simulate_knockout(dataset, model, 3, qqloss, key_sim)
    with pytest.raises(IndexError, match="Gene index -1 out of bounds for dataset"):
        simulate_knockout(dataset, model, -1, qqloss, key_sim)

    with pytest.raises(TypeError, match="Parameter `gene` must be of type str or int"):
        simulate_knockout(dataset, model, 3.14, qqloss, key_sim)  # type: ignore


@pytest.mark.parametrize("dataset_type", ["pseudotime", "markov"])
@pytest.mark.parametrize("dict_present", [False, True])
def test_simulate_all_knockouts(
    dataset_type: Literal["pseudotime", "markov"], dict_present: bool
):
    if dataset_type == "pseudotime":
        dataset = BifurcatingPseudotimeDataset(Y0)
    elif dataset_type == "markov":
        dataset = BifurcatingMarkovChainDataset(Y0)
    model = dataset.sde
    if dict_present:
        dataset.adata.uns["bowside"] = {}
        dataset.adata.uns["bowside"]["sensitivity"] = {}
    global_effects, edge_specific_effects = simulate_all_knockouts(
        dataset, model, qqloss, jax.random.key(42), loss_kwargs={"dim": 0}
    )
    adata = dataset.adata
    assert "bowside" in adata.uns, "Field 'bowside' missing from `adata.uns`."

    for gene in adata.var_names:
        assert f"simulations_ko_{gene}" in adata.uns["bowside"], (
            f"Knockout simulations for {gene} are missing."
        )

    assert "sensitivity" in adata.uns["bowside"], (
        "Sensitivity analysis results are missing."
    )

    results = adata.uns["bowside"]["sensitivity"]
    assert "global_knockout_effects" in results, "Global knockout effects are missing."
    assert global_effects.compare(results["global_knockout_effects"]).empty, (
        "Mismatch between returned global effects and those saved to adata."
    )
    assert np.all(global_effects > 0), "Knockout effects should be greater than 0."

    assert "edge_specific_knockout_effects" in results, (
        "Global knockout effects are missing."
    )
    assert edge_specific_effects.compare(
        results["edge_specific_knockout_effects"]
    ).empty, "Mismatch between returned edge-specific effects and those saved to adata."
    edge_specific_effects_np = edge_specific_effects.values.copy()
    assert np.all(np.isnan(edge_specific_effects_np.diagonal())), (
        "Effects of genes on themselves should all be NaN."
    )
    np.fill_diagonal(edge_specific_effects_np, np.inf)
    assert np.all(edge_specific_effects_np > 0), (
        "Edge-specific knockout effects should be greater than 0."
    )

    assert len(global_effects) == dataset.n_genes(), (
        "Mismatch between number of genes in dataset and length global effects."
    )
    assert edge_specific_effects.shape == (dataset.n_genes(), dataset.n_genes()), (
        "Mismatch between number of genes in dataset and shape of edge-specific effects."
    )
