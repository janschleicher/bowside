from typing import Optional, Union

import anndata as ad
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import issparse
from torch.utils.data import Dataset

from sc_markov_ode.sampling import MarkovSampler


class MarkovChainDataset(Dataset):
    """
    A pytorch dataset to generate Markov chain cell sequences, ready for
    training an ODE model.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    layer : Optional[str], optional
        Layer of the AnnData object to use as the gene expression values.
        If `None`, use adata. By default None.
    tpm_key : str, optional
        Key of the transition probability matrix in `adata.obsp` or
        `adata.uns`, by default "T_forward".
    recompute_matrix : bool, optional
        Whether the matrix should be recomputed, by default False.
    self_transitions : bool, optional
        Whether the matrix should be computed with self-transitions
        enabled, by default False.
    num_chains : int, optional
        Number of Markov chains to simulate, by default 1,000.
    max_iter : int, optional
        Maximum number of iterations for Markov chains, by default 1,000.
    rtol : float, optional
        Relative tolerance for convergence, by default 1e-5.
    atol : float, optional
        Absolute tolerance for convergence, by default 1e-5.
    n_jobs : int, optional
        Number of threads used for Markov sampling when running on CPU,
        by default -1 (all available threads).
    seed : int, optional
        Random seed, by default 42.
    copy : bool, optional
        Whether the AnnData object should be copied or modified inplace,
        by default False (modify inplace).
    """

    adata: ad.AnnData
    """AnnData object for sampling cell sequences and getting gene expression."""
    X: torch.Tensor
    """Gene expression tensor of shape (n_cells, n_genes)."""
    sampler: MarkovSampler
    """Instance of Markov chain sampler for sampling cell sequences."""
    markov_chains: np.ndarray
    """Markov chains of cell sequences, represented by their index."""

    def __init__(
        self,
        adata: ad.AnnData,
        layer: Optional[str] = None,
        mask_var: Optional[Union[NDArray[np.bool_], str]] = "highly_variable",
        tpm_key: str = "T_forward",
        recompute_matrix: bool = False,
        self_transitions: bool = False,
        num_chains: int = 1_000,
        max_iter: int = 1_000,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        n_jobs: int = -1,
        seed: int = 42,
        copy: bool = False,
    ):
        # Prepare expression data
        self.adata = adata.copy() if copy else adata
        self.X = self.adata.X if layer is None else adata.layers[layer]
        if mask_var is not None:
            if isinstance(mask_var, str):
                mask_var = self.adata.var[mask_var].values
            self.X = self.X[:, mask_var]
        if issparse(self.X):
            self.X = self.X.toarray()
        self.X = torch.from_numpy(self.X)

        # Sample Markov chains
        self.sampler = MarkovSampler(
            adata=adata,
            tpm_key=tpm_key,
            recompute_matrix=recompute_matrix,
            self_transitions=self_transitions,
            copy=False,
        )
        self.sampler.sample_markov_chains(
            num_chains=num_chains,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
            n_jobs=n_jobs,
            seed=seed,
        )
        self.markov_chains = torch.from_numpy(
            self.adata.uns["markov_sampling"]["cell_sequences"]
        )

    def __len__(self):
        return self.markov_chains.shape[0]

    def __getitem__(self, idx):
        return self.X[self.markov_chains[idx, :], :]
