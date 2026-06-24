import os
from pathlib import Path

import pytest
import scanpy as sc
import scvelo as scv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
        # Subsample to 500 cells
        sc.pp.sample(ADATA, n=500, axis="obs", rng=42)

        # Perform preprocessing
        scv.pp.filter_genes(ADATA, min_shared_counts=20)
        scv.pp.normalize_per_cell(ADATA)
        sc.pp.log1p(ADATA)
        sc.pp.highly_variable_genes(ADATA, n_top_genes=2000, subset=True)
        sc.pp.neighbors(ADATA, n_pcs=30, n_neighbors=30)
        scv.pp.moments(ADATA, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(ADATA)
        scv.tl.velocity_graph(ADATA)
        ADATA.write_h5ad(TEST_DATA_PATH)


@pytest.fixture(scope="session", autouse=True)
def make_tmp_dir():
    TMPDIR = Path(os.path.dirname(os.path.realpath(__file__))) / "tmp"
    os.environ["TMPDIR"] = str(TMPDIR)
    TMPDIR.mkdir(parents=True, exist_ok=True)
    yield
    os.removedirs(TMPDIR)
