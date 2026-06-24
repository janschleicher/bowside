"""Module for trajectory inference from a trained dynamic GRN model."""

import logging
from typing import Literal, Optional, Union

import jax
import numpy as np
from jax import numpy as jnp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.metrics import silhouette_score

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.mixture import GaussianMixtureTimeseries
from bowside.model import GRNModel, GRNModelSDE

log = logging.getLogger(__name__)


def infer_trajectories(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    key: Optional[jax.Array] = None,
    metric: Literal[
        "cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"
    ] = "euclidean",
    min_clusters: int = 2,
    max_clusters: int = 10,
    linkage_method: Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ] = "ward",
    trajectory_mode: Literal["mean", "median"] = "mean",
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
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
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
    Returns `None`. Sets the following fields in `dataset.adata.uns`:

    `adata.uns["bowside"]["simulations"]` : :class:`numpy.ndarray`
        Array of shape (n_simulations, n_steps, n_var) containing gene
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
    time_points = None
    n_steps = None
    if not isinstance(dataset, (PseudotimeDataset, MarkovChainDataset)):
        raise TypeError(
            f"Invalid dataset type: {type(dataset)}. Must be one of "
            f"[MarkovChainDataset, PseudotimeDataset]."
        )

    # Data setup
    y = dataset[:]
    y0 = jnp.asarray(y[:, 0, :].detach().numpy())

    if isinstance(dataset, PseudotimeDataset):
        time_points = dataset.time_points
    elif isinstance(dataset, MarkovChainDataset):
        n_steps = y.shape[1]

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
    if time_points is not None:
        ts = jnp.asarray(time_points.detach().cpu().numpy())
    elif n_steps is not None:
        ts = jnp.linspace(0, n_steps - 1, n_steps)

    # Sequences are of shape (n_cells, n_steps, n_genes)
    if isinstance(model, GRNModelSDE):
        assert key is not None, "For a `GRNModelSDE`, a random key is required."
        keys = jax.random.split(key, y0.shape[0])
        sequences = np.asarray(jax.vmap(model, in_axes=[None, 0, 0])(ts, y0, keys))
    else:
        sequences = np.asarray(jax.vmap(model, in_axes=[None, 0])(ts, y0))

    adata = dataset.adata
    if "bowside" not in adata.uns:
        adata.uns["bowside"] = {}
    adata.uns["bowside"]["simulations"] = sequences

    # Hierarchical clustering
    if not skip_clustering:
        distances = pdist(
            sequences.reshape(sequences.shape[0], -1),
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
        labels = np.ones(sequences.shape[0], dtype=int)
        adata.uns["bowside"]["num_trajectories"] = num_trajectories

    # Computing average expression trajectories
    trajectories = np.zeros((num_trajectories, sequences.shape[1], sequences.shape[2]))
    for i, cluster in enumerate(np.unique(labels, sorted=True)):
        cluster_sequences = sequences[labels == cluster, :, :]
        if trajectory_mode == "mean":
            trajectories[i] = cluster_sequences.mean(axis=0)
        elif trajectory_mode == "median":
            trajectories[i] = np.median(cluster_sequences, axis=0)
    adata.uns["bowside"]["trajectories"] = trajectories


def infer_cell_fates(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    score_threshold: float = 0.0,
    allow_singular_covariance: bool = False,
):
    """
    Infer the cell fate probabilities based on inferred gene expression
    trajectories.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for inferring cell fates.
    score_threshold : float, optional
        Threshold for the likelihood of a cell given the trajectories.
        Cells with a score below the threshold will not be assigned fate
        probabilities. Default: 0.0
    allow_singular_covariance : bool, optional
        Whether to allow singular covariance matrices for the Gaussian
        mixture models. Default: False.

    Returns
    -------
    Returns `None`. Sets the following fields in `dataset.adata.uns`:

    `adata.obs["bowside_cell_scores"] : :class:`pandas.Series`
        Likelihood under the Gaussian mixture model of the respective
        time point for each cell.
    `adata.obsm["bowside_fate_probabilities"] : :class:`numpy.ndarray`
        Array of shape (n_obs, n_trajectories) containing trajectory
        fate probabilities for each cell.
    `adata.obs["bowside_fate_entropy"] : :class:`pandas.Series`
        Entropy of fate probabilities for each cell. A higher value
        indicates higher differentiation potential.

    Raises
    ------
    TypeError
        An error is raised if `dataset` or `score_threshold` is of the
        wrong type.
    NotImplementedError
        An error is raised if a `MarkovChainDataset` is used. So far,
        this is only implemented for `PseudotimeDataset`.
    ValueError
        An error is raised if `score_threshold` is not between 0 and 1.
    KeyError
        An error is raised if the trajectory inference results are not
        available.
    """
    # Check dataset type
    if not isinstance(dataset, (MarkovChainDataset, PseudotimeDataset)):
        raise TypeError(
            f"Invalid dataset type: {type(dataset)}. Must be one of "
            f"[MarkovChainDataset, PseudotimeDataset]."
        )
    elif isinstance(dataset, MarkovChainDataset):
        raise NotImplementedError(
            "Cell fate inference not yet implemented for `MarkovChainDataset`."
        )

    if isinstance(score_threshold, float):
        if score_threshold < 0.0 or score_threshold > 1.0:
            raise ValueError("Parameter `score_threshold` must be between 0 and 1.")
    else:
        raise TypeError(
            f"Parameter `score_threshold` should be of type float, was {type(score_threshold)}."
        )

    # Check if trajectory inference was run
    if "bowside" not in dataset.adata.uns:
        raise KeyError(
            "Trajectory inference results not found in `dataset.adata.uns`. Run "
            "`bowside.trajectory.infer_trajectories` first."
        )

    trajectory_inference_results = dataset.adata.uns["bowside"]
    num_trajectories = trajectory_inference_results["num_trajectories"]
    if num_trajectories == 1:
        log.warning("Only one trajectory was inferred. Skipping fate prediction.")
        return

    # Create Gaussian mixture time series
    time_points = dataset.time_points.detach().cpu().numpy()
    gm_time_series = GaussianMixtureTimeseries(
        trajectory_inference_results["simulations"],
        time_points,
        trajectory_inference_results["trajectory_inference_clusters"],
        allow_singular=allow_singular_covariance,
    )
    sample_time_points = dataset.pseudotime.detach().cpu().numpy()
    samples = dataset.X.detach().cpu().numpy()
    fate_probabilities = gm_time_series.predict_proba(samples, sample_time_points)

    # Compute score for each cell given the Gaussian mixture time series
    cell_scores = np.exp(gm_time_series.score_samples(samples, sample_time_points))
    dataset.adata.obs["bowside_cell_scores"] = cell_scores
    remove_cells = cell_scores < score_threshold
    fate_probabilities[remove_cells, :] = np.nan
    dataset.adata.obsm["bowside_fate_probabilities"] = fate_probabilities
    entropies = entropy(fate_probabilities, axis=1)
    assert isinstance(entropies, np.ndarray)
    dataset.adata.obs["bowside_fate_entropy"] = entropies
