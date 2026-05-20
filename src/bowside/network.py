"""Module for network analysis on a trained dynamic GRN model."""

import logging
from collections.abc import Sequence
from typing import Dict, Literal, Optional, Union, get_args

import networkx as nx
import pandas as pd

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.model import GRNModel

log = logging.getLogger(__name__)


def _get_network(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> nx.DiGraph:
    """
    Construct a directed graph from a gene regulatory network model. If
    `network` is not None, it is returned instead of constructing a new
    graph, even if `model` is not None. In either case, this function
    checks whether the edge attributes "weight" and "dist" are present
    and computes the latter from the former if necessary.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    nx.DiGraph
        Directed graph representing the gene regulatory network, with
        edge attributes "weight" (regulation strength) and "dist"
        (distance, inverse of regulation strength, for algorithms that
        compute shortest paths).

    Raises
    ------
    ValueError
        An error is raised if both `model` and `network` are None, if
        their gene numbers do not match when both are given, or if
        `network` is given and not all edges have a weight.
    TypeError
        An error is raised if `model` or `network` is of the wrong type.
    """
    # Check types and create network if necessary
    if model is None:
        if network is None:
            raise ValueError("At least one of `model` and `network` must be set.")
        elif not isinstance(network, nx.DiGraph):
            raise TypeError("Parameter `network` must be of type `networkx.DiGraph`.")
    elif not isinstance(model, GRNModel):
        raise TypeError("Parameter `model` must be of type GRNModel.")
    else:
        if network is not None and isinstance(network, nx.DiGraph):
            if network.number_of_nodes() != model.adj.size(0):
                raise ValueError("Different number of genes for `model` and `network`.")
            log.info("Parameters `model` and `network` are both set. Using `network`.")
        else:
            idx = pd.Index(pd.Series(node_names))
            network = nx.from_pandas_adjacency(
                pd.DataFrame(
                    model.adj.detach().cpu().numpy(),
                    index=idx,
                    columns=idx,
                ),
                create_using=nx.DiGraph,
            )

    # Check for presence of weights and add distances if necessary
    add_dists = False
    for edge in network.edges(data=True):
        if "weight" not in edge[2]:
            raise ValueError("Invalid network. All edges must have a weight.")
        if "dist" not in edge[2]:
            add_dists = True
    if add_dists:
        nx.set_edge_attributes(
            network,
            {
                key: 1.0 / val
                for key, val in nx.get_edge_attributes(network, "weight").items()
            },
            name="dist",
        )

    return network


def degree_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the degree centrality for all genes in the trained gene
    regulatory network model.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Degree centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(nx.degree_centrality(network), name="degree_centrality")


def out_degree_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the outdegree centrality for all genes in the trained gene
    regulatory network model.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Outdegree centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(nx.out_degree_centrality(network), name="out_degree_centrality")


def in_degree_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the indegree centrality for all genes in the trained gene
    regulatory network model.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Indegree centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(nx.in_degree_centrality(network), name="in_degree_centrality")


def closeness_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the closeness centrality for all genes in the trained gene
    regulatory network model, using the inverse of the regulatory
    weight as the distance for shortest path calculations.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Closeness centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(
        nx.closeness_centrality(network, distance="dist"), name="closeness_centrality"
    )


def betweenness_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the betweenness centrality for all genes in the trained gene
    regulatory network model, using the inverse of the regulatory
    weight as the distance for shortest path calculations.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Betweenness centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(
        nx.betweenness_centrality(network, normalized=True, weight="dist"),
        name="betweenness_centrality",
    )


def edge_betweenness_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.DataFrame:
    """
    Compute the betweenness centrality for all edges in the trained gene
    regulatory network model, using the inverse of the regulatory
    weight as the distance for shortest path calculations.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.DataFrame
        Edge betweenness centrality for all gene pairs. A value of NaN
        for df[i, j] means that the edge from gene i to gene j does not
        exist.
    """
    network = _get_network(model, network, node_names)
    if node_names is None:
        node_names = list(network.nodes)
    return pd.Series(
        nx.edge_betweenness_centrality(network, normalized=True, weight="dist"),
        index=pd.MultiIndex.from_product([node_names, node_names]),
    ).unstack()


def eigenvector_centrality(
    model: Optional[GRNModel] = None,
    network: Optional[nx.DiGraph] = None,
    node_names: Optional[Sequence] = None,
) -> pd.Series:
    """
    Compute the closeness centrality for all genes in the trained gene
    regulatory network model, using the regulatory weight as the
    connection strength.

    Parameters
    ----------
    model : Optional[GRNModel], optional
        The trained gene regulatory network model. At least one of
        `model` and `network` must be set. Default: None.
    network : Optional[nx.DiGraph], optional
        A directed graph representing the trained gene regulatory
        network model. The graph must at least contain the edge
        attribute "weight" for all edges, optionally also the attribute
        "dist". At least one of `model` and `network` must be set.
        Default: None.
    node_names : Optional[Sequence], optional
        Node names to use when constructing the graph. Default: None
        (use integer indices as node names).

    Returns
    -------
    pd.Series
        Eigenvector centrality for all genes.
    """
    network = _get_network(model, network, node_names)
    return pd.Series(
        nx.eigenvector_centrality(network, weight="weight"),
        name="eigenvector_centrality",
    )


CentralityMeasure = Literal[
    "degree",
    "outdegree",
    "indegree",
    "closeness",
    "betweenness",
    "edge_betweenness",
    "eigenvector",
]


def compute_centrality(
    model: GRNModel,
    dataset: Optional[Union[MarkovChainDataset, PseudotimeDataset]] = None,
    measures: Union[
        Literal["all"], CentralityMeasure, Sequence[CentralityMeasure]
    ] = "all",
) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Compute different centrality measures for a trained gene regulatory
    network model.

    Parameters
    ----------
    model : GRNModel
        The trained gene regulatory network model.
    dataset : Optional[Union[MarkovChainDataset, PseudotimeDataset]], optional
        Dataset to use for gene names. If this is given, the results
        will be stored in `dataset.adata.uns`. Default: None.
    measures : Union[Literal["all"], CentralityMeasure, Sequence[CentralityMeasure]], optional
        Centrality measures to compute. Must either be "all", one of
        ["degree", "outdegree", "indegree", "closeness", "betweenness", "edge_betweenness", "eigenvector"],
        or a sequence with an arbitary combination of these measures.
        Default: "all".

    Returns
    -------
    Dict[str, Union[pd.Series, pd.DataFrame]]
        Dictionary with series for all requested node centrality
        measures and data frames for requested edge centrality measures.

    Raises
    ------
    ValueError
        An error is raised if `measure` has an invalid value.
    TypeError
        An error is raised if `dataset` is of the wrong type.
    """
    # Check centrality measures
    valid_measures = list(get_args(CentralityMeasure))
    if isinstance(measures, Sequence):
        for measure in measures:
            if measure not in valid_measures:
                raise ValueError(
                    f"{measure} is not a valid centrality measure. "
                    f"Must be one of {valid_measures}."
                )
    elif measures != "all" and measures not in valid_measures:
        raise ValueError(
            f"{measures} is not a valid centrality measure. "
            f"Must be one of {valid_measures} or 'all'."
        )
    elif measures == "all":
        measures = valid_measures
    else:
        measures = [measure]

    # Get network from model, using gene names from the dataset
    node_names = None
    if dataset is not None:
        if not isinstance(dataset, (MarkovChainDataset, PseudotimeDataset)):
            raise TypeError(
                f"Invalid dataset type: {type(dataset)}. Must be one of "
                f"[MarkovChainDataset, PseudotimeDataset]."
            )
        node_names = dataset.gene_names.tolist()
    network = _get_network(model, node_names=node_names)

    # Compute centrality measures
    results_dict = {}
    for measure in measures:
        key = f"{measure}_centrality"
        results_dict[key] = globals()[key](network=network)

    if dataset is not None:
        if "bowside" not in dataset.adata.uns:
            dataset.adata.uns["bowside"] = {}
        dataset.adata.uns["bowside"]["network"] = results_dict

    return results_dict
