import logging
import os
from typing import Literal, Optional

import numpy as np
import pytest
import scanpy as sc
import scvelo as scv
from scipy.sparse import sparray

from sc_markov_ode.utils import check_transition_matrix

TEST_DATA_PATH = "test_data/adata_velocity.h5ad"
if os.path.exists(TEST_DATA_PATH):
    ADATA = sc.read_h5ad(TEST_DATA_PATH)
else:
    ADATA = scv.datasets.pancreas(file_path="test_data/pancreas.h5ad")
    scv.pp.filter_genes(ADATA, min_shared_counts=20)
    scv.pp.normalize_per_cell(ADATA)
    scv.pp.filter_genes_dispersion(ADATA, n_top_genes=2000)
    sc.pp.log1p(ADATA)
    sc.pp.neighbors(ADATA, n_pcs=30, n_neighbors=30)
    scv.pp.moments(ADATA, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(ADATA)
    scv.tl.velocity_graph(ADATA)
    ADATA.write_h5ad(TEST_DATA_PATH)


@pytest.mark.parametrize(
    "tpm_location, terminal_states_present",
    [(None, False), ("obsp", True), ("obsp", False), ("uns", True)],
)
@pytest.mark.parametrize("recompute_matrix", [False, True])
def test_check_tpm(
    caplog,
    tpm_location: Optional[Literal["obsp", "uns"]],
    recompute_matrix: bool,
    terminal_states_present: bool,
):
    # Setup AnnData
    adata = ADATA.copy()
    tpm_key = "T_forward"
    if tpm_location == "obsp":
        adata.obsp[tpm_key] = scv.utils.get_transition_matrix(adata)
    elif tpm_location == "uns":
        adata.uns[tpm_key] = scv.utils.get_transition_matrix(adata)

    if terminal_states_present and tpm_location is not None:
        scv.tl.terminal_states(adata)

    # Perform checks
    with caplog.at_level(logging.WARNING):
        check_transition_matrix(
            adata, tpm_key=tpm_key, recompute_matrix=recompute_matrix
        )
    if tpm_location is None:
        assert (
            "Transition probability matrix was not present in .obsp or "
            ".uns and is computed from RNA velocity."
        ) in caplog.text
    assert tpm_key in adata.obsp.keys(), (
        "Transition probability matrix not present after check."
    )
    assert isinstance(adata.obsp[tpm_key], sparray), (
        "Transition probability matrix not sparse array after check."
    )
    assert np.allclose(adata.obsp[tpm_key].sum(axis=1), 1.0), (
        "Rows of the transition probability matrix do not sum to 1 after check."
    )
