import os
from pathlib import Path

import pytest
import scanpy as sc
import scvelo as scv

TEST_DATA_PATH = (
    Path(os.path.dirname(os.path.realpath(__file__))) / "test_data/adata_velocity.h5ad"
)


@pytest.fixture(scope="session", autouse=True)
def setup_adata():
    if not os.path.exists(TEST_DATA_PATH):
        ADATA = scv.datasets.pancreas(
            file_path=Path(os.path.dirname(os.path.realpath(__file__)))
            / "test_data/pancreas.h5ad"
        )
        scv.pp.filter_genes(ADATA, min_shared_counts=20)
        scv.pp.normalize_per_cell(ADATA)
        scv.pp.filter_genes_dispersion(ADATA, n_top_genes=2000)
        sc.pp.log1p(ADATA)
        sc.pp.neighbors(ADATA, n_pcs=30, n_neighbors=30)
        scv.pp.moments(ADATA, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(ADATA)
        scv.tl.velocity_graph(ADATA)
        ADATA.write_h5ad(TEST_DATA_PATH)
