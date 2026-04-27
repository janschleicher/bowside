from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scvelo as scv
import torch

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.training import integrate_model
from bowside.trajectory import infer_trajectories
from tests.conftest import TEST_DATA_PATH


class BifurcatingSDE(GRNModelSDE):
    """
    Dummy SDE generating a bifurcating system with two genes.
    """

    def __init__(self):
        super().__init__(2)
        adj = torch.tensor([[2, -2.0], [-2.0, 2.0]])
        bias = torch.tensor([1.0, 1.0])
        decay_rate = torch.tensor([0.6, 0.6])
        with torch.no_grad():
            self.adj.copy_(adj.float())
            self.log_bias.copy_(torch.log(bias.float()))
            self.log_decay_rate.copy_(torch.log(decay_rate.float()))


class BifurcatingDataset(MarkovChainDataset):
    """
    Dummy dataset simulating an SDE with a defined adjacency matrix
    instead of using a single-cell dataset.
    """

    def __init__(self, y0: torch.Tensor):
        self.sde = BifurcatingSDE()
        n_steps = 50
        self.time_points = torch.linspace(
            0, n_steps - 1, n_steps, dtype=torch.float32, device=y0.device
        )
        self.simulations = integrate_model(
            y0, self.sde, time_points=self.time_points, dt=0.1
        ).permute((1, 0, 2))
        self.adata = ad.AnnData()
        self.gene_names = pd.Index(
            [f"gene_{i}" for i in range(self.simulations.size(2))]
        )

    def __len__(self):
        return self.simulations.shape[0]

    def __getitem__(self, idx):
        return self.simulations[idx, :, :]


Y0_BALANCED = torch.tensor([[0.0, 0.0]]).repeat(500, 1)
Y0_IMBALANCED = torch.tensor([[0.0, 0.3]]).repeat(500, 1)


@pytest.mark.parametrize("y0", [Y0_BALANCED, Y0_IMBALANCED])
@pytest.mark.parametrize("trajectory_mode", ["mean", "median"])
@pytest.mark.parametrize(
    "metric, linkage_method",
    [("euclidean", "ward"), ("euclidean", "complete"), ("cosine", "average")],
)
def test_trajectory_inference(
    y0: torch.Tensor,
    trajectory_mode: Literal["mean", "median"],
    metric: Literal["cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"],
    linkage_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ],
):
    dataset = BifurcatingDataset(y0)
    model = dataset.sde
    infer_trajectories(
        dataset,
        model,
        metric=metric,
        linkage_method=linkage_method,
        trajectory_mode=trajectory_mode,
        dt=0.1,
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
        final_expression = simulations[-1, :, i]
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
                    :,
                    trajectory_inference_results["trajectory_inference_clusters"]
                    == j + 1,
                    :,
                ],
                axis=1,
            )
        ), (
            f"Trajectory coordinates for trajectory {j} are not the "
            f"{trajectory_mode} of the corresponding simulations."
        )


@pytest.mark.parametrize("min_clusters, max_clusters", [(1, 1), (2, 3), (1, 6), (2, 2)])
def test_trajectory_inference_cluster_ranges(min_clusters: int, max_clusters: int):
    dataset = BifurcatingDataset(Y0_BALANCED)
    model = dataset.sde
    infer_trajectories(
        dataset,
        model,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        dt=0.1,
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


def test_trajectory_inference_pseudotime_dataset():
    adata = sc.read_h5ad(TEST_DATA_PATH)
    scv.tl.velocity_pseudotime(adata)
    dataset = PseudotimeDataset(adata, pseudotime_key="velocity_pseudotime")
    model = GRNModelSDE(n_genes=dataset.n_genes())
    infer_trajectories(dataset, model, min_clusters=2, max_clusters=3, dt=0.01)


def test_trajectory_inference_exceptions():
    dataset = BifurcatingDataset(Y0_BALANCED)
    model = dataset.sde
    with pytest.raises(TypeError, match="Invalid dataset type"):
        infer_trajectories(dataset.adata, model, dt=0.1)  # type: ignore
    with pytest.raises(
        TypeError,
        match="Parameters `min_clusters` and `max_clusters must be of type int",
    ):
        infer_trajectories(dataset, model, min_clusters=1.2, dt=0.1)  # type: ignore
    with pytest.raises(
        TypeError,
        match="Parameters `min_clusters` and `max_clusters must be of type int",
    ):
        infer_trajectories(dataset, model, max_clusters="2", dt=0.1)  # type: ignore
    with pytest.raises(ValueError, match="Invalid range for number of clusters"):
        infer_trajectories(dataset, model, min_clusters=4, max_clusters=3, dt=0.1)
    with pytest.raises(ValueError, match="Parameter `min_clusters` must be positive."):
        infer_trajectories(dataset, model, min_clusters=-2, dt=0.1)
    with pytest.raises(
        NotImplementedError, match="Trajectory mode .+ is not implemented."
    ):
        infer_trajectories(dataset, model, trajectory_mode="max", dt=0.1)  # type: ignore
