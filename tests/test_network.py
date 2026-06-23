from collections.abc import Sequence
from typing import Optional

import equinox as eqx
import jax
import networkx as nx
import numpy as np
import pytest
from jax import numpy as jnp

from bowside.dataset import BifurcatingPseudotimeDataset
from bowside.model import GRNModel
from bowside.network import _get_network, compute_centrality

Y0 = jnp.repeat(jnp.asarray([[0.0, 0.0]]), 500, 0)


@pytest.fixture(scope="module", autouse=True)
def prepare():
    adj = jnp.asarray(
        [[1, 0, 0, -2], [2, 0, 3, 0], [0, -1, 2, 3], [0, 0, 0, 0]],
    )
    adj_init = (adj != 0).astype(int)
    global TEST_MODEL
    TEST_MODEL = GRNModel(adj_init.shape[0], adj_init, key=jax.random.key(42))
    TEST_MODEL = eqx.tree_at(lambda m: m.adj, TEST_MODEL, adj)

    global TEST_NETWORK
    TEST_NETWORK = nx.from_numpy_array(np.asarray(adj), create_using=nx.DiGraph)


@pytest.mark.parametrize(
    "use_model, use_network", [(None, True), (True, None), (True, True)]
)
@pytest.mark.parametrize("node_names", [None, ["a", "b", "c", "d"]])
def test_get_network(
    use_model: Optional[bool],
    use_network: Optional[bool],
    node_names: Optional[Sequence],
):
    model = None
    network = None
    if use_model:
        model = TEST_MODEL
    if use_network:
        network = TEST_NETWORK

    graph = _get_network(model, network, node_names)
    assert isinstance(graph, nx.DiGraph), (
        "The returned graph should be a networkx.DiGraph."
    )

    # Check that all nodes are present
    if model is not None:
        assert graph.number_of_nodes() == model.n_genes(), (
            "Mismatch between number of nodes in the graph and number of genes."
        )
    if node_names is not None and network is None:
        for node in node_names:
            assert graph.has_node(node), f"Node {node} is missing from the graph."

    # Check that all weights are positive and that distances are correct
    weights = nx.get_edge_attributes(graph, "weight", default=np.nan)
    dists = nx.get_edge_attributes(graph, "dist", default=np.nan)
    for edge, weight in weights.items():
        assert weight >= 0.0, "All weights should be positive."
        assert dists[edge] == 1.0 / weight, (
            "The distance should be the inverse of the weight."
        )


def test_get_network_exceptions():
    with pytest.raises(
        ValueError, match="At least one of `model` and `network` must be set"
    ):
        _get_network()
    with pytest.raises(
        TypeError, match="Parameter `network` must be of type `networkx.DiGraph`"
    ):
        _get_network(None, TEST_MODEL)  # type: ignore
    with pytest.raises(TypeError, match="Parameter `model` must be of type GRNModel"):
        _get_network(TEST_NETWORK, None)  # type: ignore

    graph = nx.cycle_graph(10, create_using=nx.DiGraph)
    with pytest.raises(
        ValueError, match="Invalid network. All edges must have a weight."
    ):
        _get_network(None, graph)

    nx.set_edge_attributes(graph, {edge: 1.0 for edge in graph.edges}, name="weight")
    with pytest.raises(
        ValueError, match="Different number of genes for `model` and `network`"
    ):
        _get_network(TEST_MODEL, graph)


@pytest.mark.parametrize("use_dataset", [False, True])
@pytest.mark.parametrize(
    "measures",
    [
        "all",
        "degree",
        ["degree", "betweenness"],
        ["closeness", "betweenness", "edge_betweenness"],
    ],
)
def test_compute_centrality(use_dataset: bool, measures):
    if use_dataset:
        dataset = BifurcatingPseudotimeDataset(Y0)
        model = dataset.sde
    else:
        dataset = None
        model = TEST_MODEL
    compute_centrality(model, dataset, measures)


def test_compute_centrality_exceptions():
    with pytest.raises(
        ValueError,
        match=" is not a valid centrality measure. Must be one of .* or 'all'",
    ):
        compute_centrality(TEST_MODEL, measures="bogus")  # type: ignore
    with pytest.raises(
        ValueError, match=" is not a valid centrality measure. Must be one of "
    ):
        compute_centrality(TEST_MODEL, measures=["more", "bogus"])  # type: ignore
    with pytest.raises(TypeError, match="Invalid dataset type"):
        compute_centrality(TEST_MODEL, np.ones((4, 4)))  # type: ignore
