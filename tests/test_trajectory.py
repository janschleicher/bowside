from typing import Literal

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from bowside.dataset import BifurcatingMarkovChainDataset, BifurcatingPseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.trajectory import infer_cell_fates, infer_trajectories

Y0_BALANCED = jnp.repeat(jnp.asarray([[0.0, 0.0]]), 500, 0)
Y0_IMBALANCED = jnp.repeat(jnp.asarray([[0.0, 0.3]]), 500, 0)


@pytest.mark.parametrize("y0", [Y0_BALANCED, Y0_IMBALANCED])
@pytest.mark.parametrize("trajectory_mode", ["mean", "median"])
@pytest.mark.parametrize(
    "metric, linkage_method",
    [("euclidean", "ward"), ("euclidean", "complete"), ("cosine", "average")],
)
def test_trajectory_inference(
    y0: jax.Array,
    trajectory_mode: Literal["mean", "median"],
    metric: Literal["cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"],
    linkage_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ],
):
    dataset = BifurcatingPseudotimeDataset(y0)
    model = dataset.sde
    infer_trajectories(
        dataset,
        model,
        key=jax.random.key(42),
        metric=metric,
        linkage_method=linkage_method,
        trajectory_mode=trajectory_mode,
    )
    adata = dataset.adata
    assert "bowside" in adata.uns, "Field 'bowside' missing from `adata.uns`."

    trajectory_inference_results = adata.uns["bowside"]
    assert "simulations" in trajectory_inference_results, (
        "Simulations are missing from trajectory inference results."
    )
    assert "silhouette_scores" in trajectory_inference_results, (
        "Silhouette scores are missing from trajectory inference results."
    )
    assert "num_trajectories" in trajectory_inference_results, (
        " Number of trajectories is missing from trajectory inference results."
    )
    assert "trajectory_inference_linkage" in trajectory_inference_results, (
        "Linkage tree is missing from trajectory inference results."
    )
    assert "trajectory_inference_clusters" in trajectory_inference_results, (
        "Cluster labels are missing from trajectory inference results."
    )
    assert "trajectories" in trajectory_inference_results, (
        "Trajectories are missing from trajectory inference results."
    )

    num_trajectories = trajectory_inference_results["num_trajectories"]
    simulations = trajectory_inference_results["simulations"]

    assert num_trajectories == 2, (
        f"There should be 2 trajectories, but there are {num_trajectories}."
    )

    for i in range(dataset.n_genes()):
        gene_i_up_trajectory = trajectory_inference_results["trajectories"][
            :, -1, i
        ].argmax()
        final_expression = simulations[:, -1, i]
        cluster_indices = (
            trajectory_inference_results["trajectory_inference_clusters"]
            == gene_i_up_trajectory + 1
        )
        assert np.quantile(final_expression[cluster_indices], 0.2) >= np.quantile(
            final_expression[~cluster_indices], 0.8
        ), (
            f"The final expression of gene {i} should be higher in simulations "
            "of its upregulated trajectory than in the other simulations."
        )

    for j, trajectory in enumerate(trajectory_inference_results["trajectories"]):
        agg_func = np.mean if trajectory_mode == "mean" else np.median
        assert np.all(
            trajectory
            == agg_func(
                simulations[
                    trajectory_inference_results["trajectory_inference_clusters"]
                    == j + 1,
                    :,
                    :,
                ],
                axis=0,
            )
        ), (
            f"Trajectory coordinates for trajectory {j} are not the "
            f"{trajectory_mode} of the corresponding simulations."
        )


# FIXME: fails with EquinoxRuntimeError -> maximum number of solver steps was reached
@pytest.mark.parametrize("min_clusters, max_clusters", [(1, 1), (2, 3), (1, 6), (2, 2)])
def test_trajectory_inference_cluster_ranges(min_clusters: int, max_clusters: int):
    dataset = BifurcatingMarkovChainDataset(Y0_BALANCED)
    model = dataset.sde
    infer_trajectories(
        dataset,
        model,
        key=jax.random.key(42),
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    adata = dataset.adata
    assert "bowside" in adata.uns, "Field 'bowside' missing from `adata.uns`."

    trajectory_inference_results = adata.uns["bowside"]
    assert "simulations" in trajectory_inference_results, (
        "Simulations are missing from trajectory inference results."
    )
    assert "num_trajectories" in trajectory_inference_results, (
        " Number of trajectories is missing from trajectory inference results."
    )
    assert "trajectories" in trajectory_inference_results, (
        "Trajectories are missing from trajectory inference results."
    )

    num_clusterings = max_clusters - min_clusters + 1
    if min_clusters == 1 and max_clusters >= 1:
        num_clusterings -= 1

    if min_clusters == 1 and max_clusters == 1:
        assert (
            "silhouette_scores" not in trajectory_inference_results
            and "trajectory_inference_linkage" not in trajectory_inference_results
            and "trajectory_inference_clusters" not in trajectory_inference_results
        ), (
            "With `min_clusters` and `max_clusters` both set to 1, no clustering should be performed."
        )
        assert trajectory_inference_results["num_trajectories"] == 1, (
            "There should only be one trajectory."
        )
    else:
        assert "silhouette_scores" in trajectory_inference_results, (
            "Silhouette scores are missing from trajectory inference results."
        )
        assert (
            len(trajectory_inference_results["silhouette_scores"]) == num_clusterings
        ), "Wrong number of tested numbers of trajectories."

        assert "trajectory_inference_linkage" in trajectory_inference_results, (
            "Linkage tree is missing from trajectory inference results."
        )
        assert "trajectory_inference_clusters" in trajectory_inference_results, (
            "Cluster labels are missing from trajectory inference results."
        )
        assert (
            trajectory_inference_results["num_trajectories"] >= min_clusters
            and trajectory_inference_results["num_trajectories"] <= max_clusters
        ), (
            f"The number of trajectories should be between {min_clusters} and {max_clusters}"
        )

    assert (
        trajectory_inference_results["trajectories"].shape[0]
        == trajectory_inference_results["num_trajectories"]
    ), "Mismatch between number of trajectories and trajectory coordinates array shape."


def test_trajectory_inference_markov_chain_dataset():
    dataset = BifurcatingMarkovChainDataset(Y0_BALANCED)
    model = GRNModelSDE(n_genes=dataset.n_genes(), key=jax.random.key(42), dt=0.1)
    infer_trajectories(
        dataset, model, key=jax.random.key(97), min_clusters=2, max_clusters=3
    )


def test_trajectory_inference_exceptions():
    dataset = BifurcatingPseudotimeDataset(Y0_BALANCED)
    model = dataset.sde
    with pytest.raises(TypeError, match="Invalid dataset type"):
        infer_trajectories(dataset.adata, model, key=jax.random.key(42))  # type: ignore
    with pytest.raises(
        TypeError,
        match="Parameters `min_clusters` and `max_clusters must be of type int",
    ):
        infer_trajectories(dataset, model, key=jax.random.key(42), min_clusters=1.2)  # type: ignore
    with pytest.raises(
        TypeError,
        match="Parameters `min_clusters` and `max_clusters must be of type int",
    ):
        infer_trajectories(dataset, model, key=jax.random.key(42), max_clusters="2")  # type: ignore
    with pytest.raises(ValueError, match="Invalid range for number of clusters"):
        infer_trajectories(
            dataset, model, key=jax.random.key(42), min_clusters=4, max_clusters=3
        )
    with pytest.raises(ValueError, match="Parameter `min_clusters` must be positive."):
        infer_trajectories(dataset, model, key=jax.random.key(42), min_clusters=-2)
    with pytest.raises(
        NotImplementedError, match="Trajectory mode .+ is not implemented."
    ):
        infer_trajectories(
            dataset,
            model,
            key=jax.random.key(42),
            trajectory_mode="max",  # type: ignore
        )


@pytest.mark.parametrize(
    "n_clusters, score_threshold",
    [(2, 0.0), (1, 0.0), (3, 0.0), (2, 0.95)],
)
def test_fate_inference(n_clusters: int, score_threshold: float):
    dataset = BifurcatingPseudotimeDataset(Y0_BALANCED)
    model = dataset.sde
    infer_trajectories(
        dataset,
        model,
        key=jax.random.key(42),
        min_clusters=n_clusters,
        max_clusters=n_clusters,
    )
    infer_cell_fates(dataset, score_threshold=score_threshold)

    # Check presence/absence of results, depending on number of trajectories
    if n_clusters == 1:
        assert "bowside_cell_scores" not in dataset.adata.obs.columns, (
            "Cell should not be scored with only 1 trajectory."
        )
        assert "bowside_fate_probabilities" not in dataset.adata.obsm, (
            "Fate probabilities should not be computed with only 1 trajectory."
        )
        assert "bowside_fate_entropy" not in dataset.adata.obs.columns, (
            "Fate entropies should not be computed with only 1 trajectory."
        )
        return
    assert "bowside_cell_scores" in dataset.adata.obs.columns, (
        "Cell scores are missing."
    )
    assert "bowside_fate_probabilities" in dataset.adata.obsm, (
        "Fate probabilities are missing."
    )
    assert "bowside_fate_entropy" in dataset.adata.obs.columns, (
        "Fate entropies are missing."
    )

    # Check inferred probabilities
    probabilities = dataset.adata.obsm["bowside_fate_probabilities"]
    assert probabilities.shape == (len(dataset.adata), n_clusters), (
        "Probabilities have wrong shape."
    )
    unassigned_cells = np.isnan(probabilities).any(axis=1)
    assert np.allclose(np.sum(probabilities[~unassigned_cells, :], axis=1), 1.0), (
        "Probabilities should sum to 1 for all cells."
    )
    assert (unassigned_cells.sum() > 0) == (score_threshold > 0.9), (
        f"There schould{' not' if score_threshold == 0 else ''} be "
        f"unassigned cells at threshold {score_threshold}."
    )


def test_fate_inference_exceptions():
    dataset = BifurcatingMarkovChainDataset(Y0_BALANCED)
    with pytest.raises(
        NotImplementedError,
        match="Cell fate inference not yet implemented for `MarkovChainDataset`.",
    ):
        infer_cell_fates(dataset)

    dataset = BifurcatingPseudotimeDataset(Y0_BALANCED)
    with pytest.raises(TypeError, match="Invalid dataset type:"):
        infer_cell_fates(dataset.adata)  # type: ignore

    with pytest.raises(
        KeyError,
        match="Trajectory inference results not found in `dataset.adata.uns`. Run "
        "`bowside.trajectory.infer_trajectories` first.",
    ):
        infer_cell_fates(dataset)

    dataset.adata.uns["bowside"] = {}
    with pytest.raises(
        ValueError, match="Parameter `score_threshold` must be between 0 and 1."
    ):
        infer_cell_fates(dataset, score_threshold=1.5)
    with pytest.raises(
        ValueError, match="Parameter `score_threshold` must be between 0 and 1."
    ):
        infer_cell_fates(dataset, score_threshold=-0.2)
    with pytest.raises(
        TypeError, match="Parameter `score_threshold` should be of type float"
    ):
        infer_cell_fates(dataset, score_threshold="0.2")  # type: ignore
