import logging
import re

import numpy as np
import pytest
import scanpy as sc
import torch

from bowside.sampling import MarkovSampler
from tests.conftest import TEST_DATA_PATH


@pytest.mark.parametrize(
    "num_chains, max_iter, copy",
    [
        (1_000, 100, False),
        (1_000, 1_000, False),
        (2_000, 1_000, False),
        (1_000, 1_000, True),
    ],
)
def test_markov_chain_sampling(num_chains: int, max_iter: int, copy: bool):
    adata_orig = sc.read_h5ad(TEST_DATA_PATH)
    sampler = MarkovSampler(adata_orig, copy=copy)
    return_val = sampler.sample_markov_chains(num_chains=num_chains, max_iter=max_iter)
    if copy:
        assert isinstance(return_val, sc.AnnData), (
            "When `copy=True`, an AnnData object should be returned."
        )
        adata = return_val
    else:
        assert return_val is None, "When `copy=False`, nothing should be returned."
        adata = sampler.adata

    assert adata.uns["markov_sampling"], (
        "The `AnnData` object should contain the sampling results."
    )
    cell_sequences_max_iter = adata.uns["markov_sampling"]["cell_sequences_max_iter"]
    assert cell_sequences_max_iter.shape == (
        num_chains,
        max_iter,
    ), (
        f"Shape of cell_sequences_max_iter should be ({num_chains}, {max_iter}); actual: "
        f"{cell_sequences_max_iter.shape}."
    )
    cell_sequences = adata.uns["markov_sampling"]["cell_sequences"]
    convergence = adata.uns["markov_sampling"]["convergence"]
    assert cell_sequences.shape == (num_chains, convergence), (
        f"Shape of cell_sequences should be ({num_chains}, {convergence}); actual: "
        f"{cell_sequences.shape}."
    )
    assert np.all(cell_sequences == cell_sequences_max_iter[:, :convergence]), (
        "Mismatch between cell_sequences and cell_sequences_max_iter up to convergence."
    )
    assert torch.all(sampler.p_init[cell_sequences[:, 0]] > 0), (
        "Initial state probability of cell sequence starting points should always be > 0."
    )


@pytest.mark.parametrize("max_iter", [10, 1_000])
def test_stationary_distribution(caplog, max_iter: int):
    sampler = MarkovSampler(sc.read_h5ad(TEST_DATA_PATH))
    caplog.clear()
    assert torch.allclose(
        sampler.p_init.sum(), torch.tensor(1.0, dtype=torch.float32)
    ), "The initial distribution should sum to 1."

    with caplog.at_level(logging.INFO):
        sampler._compute_stationary_distribution(max_iter=max_iter)
    if max_iter < 50:
        assert caplog.records[0].levelname == "WARNING", (
            "A warning should be issued when the stationary distribution is not found."
        )
        assert (
            f"The Markov chain did not converge to the stationary distribution "
            f"after {max_iter} steps"
        ) in caplog.text
        assert sampler.convergence == max_iter, (
            "If the stationary distribution is not found, convergence should be the same as max_iter."
        )
    else:
        assert caplog.records[0].levelname == "INFO", (
            "An info statement should be issued, containing the number of steps to convergence."
        )
        assert re.search(
            r"Convergence to stationary distribution reached after \d+ steps\.",
            caplog.text,
        )
        assert sampler.convergence < max_iter, (
            "The number of steps to convergence should be smaller than max_iter."
        )

    assert torch.allclose(
        sampler.p_term.sum(), torch.tensor(1.0, dtype=torch.float32)
    ), "The stationary distribution should sum to 1."
