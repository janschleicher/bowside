import logging
from typing import Literal, Optional, Union

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.model import GRNModel
from bowside.training import integrate_model

log = logging.getLogger(__name__)


def infer_trajectories(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    metric: Literal[
        "cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"
    ] = "euclidean",
    min_clusters: int = 2,
    max_clusters: int = 10,
    linkage_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "ward",
    trajectory_mode: Literal["mean", "median"] = "mean",
    solver_method: Optional[str] = None,
    dt: float = 0.001,
    atol: float = 1e-4,
    rtol: float = 1e-5,
):
    """
    Infer average gene expression trajectories from differential equation
    model simulations through hierarchical clustering.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for simulations (starting cells and time points).
    model : GRNModel
        The trained gene regulatory network model.
    metric : Literal["cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"], optional
        Distance metric for clustering of simulations. Default: "euclidean".
    min_clusters : int, optional
        Minimum number of clusters for automatic identification of the
        best number of trajectories. Default: 2.
    max_clusters : int, optional
        Maximum number of clusters for automatic identification of the
        best number of trajectories. Default: 10.
    linkage_method : Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"], optional
        Linkage algorithm for hierarchical clustering. Refer to
        :func:`scipy.cluster.hierarchy.linkage` documentation for details.
        Methods "centroid", "median", and "ward" expect Euclidean distances.
        Default: "ward".
    trajectory_mode : Literal["mean", "median"], optional
        Mode for averaging gene expression trajectories per cluster.
        Default: "mean".
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the model type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.

    Returns
    -------
    Returns `None`. Sets the following fields in `adata.uns`:

    `adata.uns["bowside"]["simulations"]` : :class:`numpy.ndarray`
        Array of shape (n_steps, n_simulations, n_var) containing gene
        expression simulations from the trained model.
    `adata.uns["bowside"]["num_trajectories"]` : int
        Number of inferred trajectories. If `min_clusters < max_clusters`,
        this will be the optimal number of trajectories according to
        silhouette score.
    `adata.uns["bowside"]["silhouette_scores"]` : Dict[int, float]
        Dictionary of silhouette scores for different numbers of
        trajectories. This is not set when both `min_clusters` and
        `max_clusters` are 1.
    `adata.uns["bowside"]["trajectory_inference_linkage"]` : :class:`numpy.ndarray`
        Linkage matrix representing the hierarchical clustering of the
        simulations.
    `adata.uns["bowside"]["trajectory_inference_clusters"]` : :class:`numpy.ndarray`
        An array of shape (n_simulations). The value at index i is the
        cluster number to which simulation i belongs.

    Raises
    ------
    TypeError
        An error is raised if an invalid type of dataset is passed, or
        if `min_clusters` or `max_clusters` are not of type int.
    ValueError
        An error is raised if an invalid range is given for the number
        of clkusters or if the minimum number of clusters is smaller
        than 1.
    NotImplementedError
        An error is raised if an invalid value for `trajectory_mode` is
        passed.
    """
    # Data setup
    y = dataset[:]
    y0 = y[:, 0, :]

    time_points = None
    n_steps = None
    if isinstance(dataset, PseudotimeDataset):
        time_points = dataset.time_points
    elif isinstance(dataset, MarkovChainDataset):
        n_steps = y.shape[1]
    else:
        raise TypeError(
            f"Invalid dataset type: {type(dataset)}. Must be one of "
            f"[MarkovChainDataset, PseudotimeDataset]."
        )

    # Argument checks
    skip_clustering = False
    if not isinstance(min_clusters, int) or not isinstance(max_clusters, int):
        raise TypeError(
            f"Parameters `min_clusters` and `max_clusters must be of "
            f"type int, but were {type(min_clusters)} and {type(max_clusters)}"
        )
    elif max_clusters < min_clusters:
        raise ValueError(
            f"Invalid range for number of clusters: min_clusters="
            f"{min_clusters}, max_clusters={max_clusters}."
        )
    elif min_clusters < 2:
        if min_clusters < 1:
            raise ValueError("Parameter `min_clusters` must be positive.")
        elif max_clusters != min_clusters:
            log.info(
                "Cannot compute silhouette score for one cluster. Setting"
                "`min_clusters` to 2 for choosing the best number of clusters."
            )
            min_clusters = 2
        else:
            log.info(
                "Skipping clustering, only generating mean trajectory "
                "because `min_clusters` and `max_clusters` are both 1."
            )
            skip_clustering = True
    if trajectory_mode not in ["mean", "median"]:
        raise NotImplementedError(
            f"Trajectory mode {trajectory_mode} is not implemented."
        )

    # Simulate sequences with differential equation model
    model.eval()
    with torch.no_grad():
        sequences = (
            integrate_model(
                y0,
                model,
                n_steps=n_steps,
                time_points=time_points,
                adjoint=False,
                solver_method=solver_method,
                dt=dt,
                atol=atol,
                rtol=rtol,
            )
            .detach()
            .cpu()
            .numpy()
        )

    adata = dataset.adata
    if "bowside" not in adata.uns:
        adata.uns["bowside"] = {}
    adata.uns["bowside"]["simulations"] = sequences

    # Hierarchical clustering
    if not skip_clustering:
        distances = pdist(
            np.moveaxis(sequences, 1, 0).reshape(sequences.shape[1], -1),
            metric=metric,
        )
        distances_square = squareform(distances)
        linkage_matrix = linkage(distances, method=linkage_method)

        # Find best number of clusters
        silhouette_scores = {}
        cluster_labels = {}
        for n_clusters in range(min_clusters, max_clusters + 1):
            cluster_labels[n_clusters] = fcluster(
                linkage_matrix, t=n_clusters, criterion="maxclust"
            )
            silhouette_scores[n_clusters] = silhouette_score(
                distances_square, cluster_labels[n_clusters], metric="precomputed"
            )
        num_trajectories = max(silhouette_scores, key=lambda x: silhouette_scores[x])
        labels = cluster_labels[num_trajectories]
        adata.uns["bowside"]["silhouette_scores"] = silhouette_scores
        adata.uns["bowside"]["num_trajectories"] = num_trajectories
        adata.uns["bowside"]["trajectory_inference_linkage"] = linkage_matrix
        adata.uns["bowside"]["trajectory_inference_clusters"] = labels
    else:
        num_trajectories = 1
        labels = np.ones(sequences.shape[1], dtype=int)
        adata.uns["bowside"]["num_trajectories"] = num_trajectories

    # Computing average expression trajectories
    trajectories = np.zeros((num_trajectories, sequences.shape[0], sequences.shape[2]))
    for i, cluster in enumerate(np.unique(labels, sorted=True)):
        cluster_sequences = sequences[:, labels == cluster, :]
        if trajectory_mode == "mean":
            trajectories[i] = cluster_sequences.mean(axis=1)
        elif trajectory_mode == "median":
            trajectories[i] = np.median(cluster_sequences, axis=1)
    adata.uns["bowside"]["trajectories"] = trajectories
