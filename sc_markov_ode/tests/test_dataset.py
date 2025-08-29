from typing import Optional, Union

import numpy as np
import pytest
import scanpy as sc
import scvelo as scv
from numpy.typing import NDArray

from sc_markov_ode.dataset import MarkovChainDataset
from sc_markov_ode.tests.conftest import TEST_DATA_PATH


@pytest.fixture(scope="module", autouse=True)
def prepare():
    global ADATA
    ADATA = sc.read_h5ad(TEST_DATA_PATH)
    ADATA.obsp["T_forward"] = scv.utils.get_transition_matrix(
        ADATA, self_transitions=False
    )
    scv.tl.terminal_states(ADATA, self_transitions=False)


@pytest.mark.parametrize(
    "mask_var",
    [None, "highly_variable", "array"],
)
def test_dataset(mask_var: Optional[Union[NDArray[np.bool_], str]]):
    adata = ADATA.copy()
    if mask_var == "array":
        rng = np.random.default_rng(42)
        mask_var = rng.choice(2, adata.shape[1]).astype(bool)
    num_chains = 1_000
    dataset = MarkovChainDataset(adata, mask_var=mask_var, num_chains=num_chains)

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
