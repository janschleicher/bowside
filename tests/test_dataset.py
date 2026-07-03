from contextlib import nullcontext as does_not_raise
from math import ceil, floor
from typing import Optional, Union

import numpy as np
import pytest
import scanpy as sc
import scvelo as scv
import torch
from numpy.typing import NDArray

from bowside.dataset import (
    MarkovChainDataset,
    PseudotimeDataset,
    PseudotimeTimepointWrapper,
    VelocityDataset,
    split_by_time_point,
)
from tests.conftest import TEST_DATA_PATH


@pytest.fixture(scope="module", autouse=True)
def prepare():
    global ADATA
    ADATA = sc.read_h5ad(TEST_DATA_PATH)
    ADATA.obsp["T_forward"] = scv.utils.get_transition_matrix(
        ADATA, self_transitions=False
    ).toarray()
    scv.tl.terminal_states(ADATA, self_transitions=False)
    scv.tl.velocity_pseudotime(ADATA)


@pytest.mark.parametrize(
    "mask_var, expectation",
    [
        (None, does_not_raise()),
        ("highly_variable", does_not_raise()),
        ("array", does_not_raise()),
        (
            "means",
            pytest.raises(
                TypeError,
                match=r"Expected adata\.var\[mask_var\] to be of type bool, but got type \w+",
            ),
        ),
    ],
)
@pytest.mark.parametrize("layer", [None, "spliced"])
def test_markov_chain_dataset(
    mask_var: Optional[Union[NDArray[np.bool_], str]], layer: Optional[str], expectation
):
    adata = ADATA.copy()
    if mask_var == "array":
        rng = np.random.default_rng(42)
        mask_var = rng.choice(2, adata.shape[1]).astype(bool)
    num_chains = 1_000
    with expectation as context:
        dataset = MarkovChainDataset(
            adata, layer=layer, mask_var=mask_var, num_chains=num_chains
        )
    if context is not None:
        return

    if isinstance(mask_var, np.ndarray):
        assert dataset.X.shape == (adata.shape[0], mask_var.sum()), (
            "Shape mismatch of AnnData and X when using boolean mask."
        )
    elif mask_var == "highly_variable":
        assert dataset.X.shape == (
            adata.shape[0],
            adata.var["highly_variable"].sum(),
        ), "Shape mismatch of AnnData and X when using highly variable genes."
    else:
        assert dataset.X.shape == adata.shape, "Shape mismatch of AnnData and X."

    batch_size = 4
    minibatch = dataset[:4]
    assert minibatch.shape == (
        batch_size,
        dataset.markov_chains.shape[1],
        dataset.X.shape[1],
    ), "Shape mismatch of minibatch and (batch_size, num_steps, num_genes)."
    assert len(dataset) == num_chains, "Dataset has incorrect length."


@pytest.mark.parametrize(
    "mask_var, pseudotime_key, expectation",
    [
        (None, "velocity_pseudotime", does_not_raise()),
        (
            None,
            "pseudotime",
            pytest.raises(KeyError, match="Pseudotime key not found in adata.obs."),
        ),
        ("highly_variable", "velocity_pseudotime", does_not_raise()),
        ("array", "velocity_pseudotime", does_not_raise()),
        (
            "means",
            "velocity_pseudotime",
            pytest.raises(
                TypeError,
                match=r"Expected adata\.var\[mask_var\] to be of type bool, but got type \w+",
            ),
        ),
    ],
)
@pytest.mark.parametrize("layer", [None, "Ms"])
def test_pseudotime_dataset(
    mask_var: Optional[Union[NDArray[np.bool_], str]],
    layer: Optional[str],
    pseudotime_key: str,
    expectation,
):
    adata = ADATA.copy()
    if mask_var == "array":
        rng = np.random.default_rng(42)
        mask_var = rng.choice(2, adata.shape[1]).astype(bool)
    with expectation as context:
        dataset = PseudotimeDataset(
            adata,
            layer=layer,
            mask_var=mask_var,
            pseudotime_key=pseudotime_key,
        )
    if context is not None:
        return

    if isinstance(mask_var, np.ndarray):
        assert dataset.X.shape == (adata.shape[0], mask_var.sum()), (
            "Shape mismatch of AnnData and X when using boolean mask."
        )
    elif mask_var == "highly_variable":
        assert dataset.X.shape == (
            adata.shape[0],
            adata.var["highly_variable"].sum(),
        ), "Shape mismatch of AnnData and X when using highly variable genes."
    else:
        assert dataset.X.shape == adata.shape, "Shape mismatch of AnnData and X."

    batch_size = 4
    minibatch_X = dataset[:4]
    assert minibatch_X.shape == (
        batch_size,
        dataset.sampled_cells.shape[1],
        dataset.X.shape[1],
    ), "Shape mismatch of minibatch and (batch_size, num_genes) for expression matrix."


@pytest.mark.parametrize(
    "mask_var, expectation",
    [
        (None, does_not_raise()),
        ("highly_variable", does_not_raise()),
        ("array", does_not_raise()),
        (
            "means",
            pytest.raises(
                TypeError,
                match=r"Expected adata\.var\[mask_var\] to be of type bool, but got type \w+",
            ),
        ),
    ],
)
@pytest.mark.parametrize("layer", [None, "Ms"])
def test_velocity_dataset(
    mask_var: Optional[Union[NDArray[np.bool_], str]], layer: Optional[str], expectation
):
    adata = ADATA.copy()
    if mask_var == "array":
        rng = np.random.default_rng(42)
        mask_var = rng.choice(2, adata.shape[1]).astype(bool)
    with expectation as context:
        dataset = VelocityDataset(adata, layer=layer, mask_var=mask_var)
    if context is not None:
        return

    if isinstance(mask_var, np.ndarray):
        assert dataset.X.shape == (adata.shape[0], mask_var.sum()), (
            "Shape mismatch of AnnData and X when using boolean mask."
        )
    elif mask_var == "highly_variable":
        assert dataset.X.shape == (
            adata.shape[0],
            adata.var["highly_variable"].sum(),
        ), "Shape mismatch of AnnData and X when using highly variable genes."
    else:
        assert dataset.X.shape == adata.shape, "Shape mismatch of AnnData and X."

    batch_size = 4
    minibatch_X, minibatch_V = dataset[:4]
    assert minibatch_X.shape == (
        batch_size,
        dataset.X.shape[1],
    ), "Shape mismatch of minibatch and (batch_size, num_genes) for expression matrix."
    assert minibatch_V.shape == (
        batch_size,
        dataset.V.shape[1],
    ), "Shape mismatch of minibatch and (batch_size, num_genes) for velocity matrix."


@pytest.mark.parametrize("n_time_points", [20, 21, 100])
@pytest.mark.parametrize(
    "train_proportion, validation_proportion, test_proportion",
    [(0.8, 0.2, 0.0), (0.6, 0.2, 0.2), (0.7, 0.1, 0.2)],
)
def test_split_by_time_point(
    n_time_points: int,
    train_proportion: float,
    validation_proportion: float,
    test_proportion: float,
):
    n_cells = 100
    adata = ADATA.copy()
    dataset = PseudotimeDataset(
        adata,
        num_cells=n_cells,
        num_quantiles=n_time_points,
        mask_var="highly_variable",
        pseudotime_key="velocity_pseudotime",
    )
    indices_tuple = split_by_time_point(
        dataset,
        train_proportion=train_proportion,
        validation_proportion=validation_proportion,
        test_proportion=test_proportion,
    )
    test_indices = None
    test_dataset = None
    if len(indices_tuple) == 2:
        train_indices, validation_indices = indices_tuple
    else:
        train_indices, validation_indices, test_indices = indices_tuple
        expected_n_test = floor(test_proportion * n_time_points)
        assert len(test_indices) == expected_n_test, (
            f"Wrong number of time points for test. Expected: {expected_n_test},"
            f"actual: {len(test_indices)}."
        )
        test_dataset = PseudotimeTimepointWrapper(dataset, test_indices)
        assert test_dataset[:].shape == (n_cells, expected_n_test, dataset.n_genes()), (
            "Shape mismatch for test dataset."
        )

    expected_n_train = ceil(train_proportion * n_time_points)
    assert len(train_indices) == expected_n_train, (
        f"Wrong number of time points for training. Expected: {expected_n_train},"
        f"actual: {len(train_indices)}."
    )

    expected_n_validation = floor(validation_proportion * n_time_points)
    assert len(validation_indices) == expected_n_validation, (
        f"Wrong number of time points for validation. Expected: {expected_n_validation},"
        f"actual: {len(validation_indices)}."
    )

    train_dataset = PseudotimeTimepointWrapper(dataset, train_indices)
    assert train_dataset[:].shape == (n_cells, expected_n_train, dataset.n_genes()), (
        "Shape mismatch for training dataset."
    )
    validation_dataset = PseudotimeTimepointWrapper(dataset, validation_indices)
    assert validation_dataset[:].shape == (
        n_cells,
        expected_n_validation,
        dataset.n_genes(),
    ), "Shape mismatch for validation dataset."

    assert torch.min(validation_dataset.time_points) > torch.max(
        train_dataset.time_points
    ), "The validation set should have higher pseudotimes than the train set."
    if test_dataset is not None:
        assert torch.min(validation_dataset.time_points) > torch.max(
            test_dataset.time_points
        ), "The validation set should have higher pseudotimes than the test set."


def test_split_by_time_point_exceptions():
    adata = ADATA.copy()
    dataset = PseudotimeDataset(
        adata,
        num_cells=100,
        num_quantiles=20,
        mask_var="highly_variable",
        pseudotime_key="velocity_pseudotime",
    )
    with pytest.raises(ValueError, match="Proportions must sum up to 1."):
        split_by_time_point(
            dataset,
            train_proportion=0.8,
            validation_proportion=0.2,
            test_proportion=0.2,
        )
    with pytest.raises(ValueError, match="Proportions must sum up to 1."):
        split_by_time_point(
            dataset,
            train_proportion=0.5,
            validation_proportion=0.2,
            test_proportion=0.2,
        )

    with pytest.raises(
        ValueError, match="Training proportion must be in \\(0, 1\\), but was"
    ):
        split_by_time_point(
            dataset,
            train_proportion=1.1,
            validation_proportion=-0.1,
            test_proportion=0.0,
        )
    with pytest.raises(
        ValueError, match="Training proportion must be in \\(0, 1\\), but was"
    ):
        split_by_time_point(
            dataset,
            train_proportion=1.0,
            validation_proportion=0.0,
            test_proportion=0.0,
        )
    with pytest.raises(
        ValueError, match="Validation proportion must be in \\(0, 1\\), but was"
    ):
        split_by_time_point(
            dataset,
            train_proportion=0.8,
            validation_proportion=-0.1,
            test_proportion=0.3,
        )
    with pytest.raises(
        ValueError, match="Validation proportion must be in \\(0, 1\\), but was"
    ):
        split_by_time_point(
            dataset,
            train_proportion=0.8,
            validation_proportion=-0.2,
            test_proportion=0.4,
        )
    with pytest.raises(
        ValueError, match="Test proportion must be in \\[0, 1\\), but was"
    ):
        split_by_time_point(
            dataset,
            train_proportion=0.8,
            validation_proportion=0.3,
            test_proportion=-0.1,
        )
