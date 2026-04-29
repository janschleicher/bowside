"""Module for datasets to be used to train an ODE GRN model."""

from collections.abc import Sequence
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import sparse
from torch.utils.data import Dataset

from bowside.model import BifurcatingSDE
from bowside.sampling import MarkovSampler


class MarkovChainDataset(Dataset):
    """
    A pytorch dataset to generate Markov chain cell sequences, ready for
    training an ODE or SDE model.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    layer : Optional[str], optional
        Layer of the AnnData object to use as the gene expression values.
        If `None`, use adata.X. Default: None.
    mask_var : Optional[Union[NDArray[np.bool_], str]], optional
        Only use a subset of genes given by a boolean array or a string
        referring to a column in `adata.var`. Default: None (use
        all genes).
    tpm_key : str, optional
        Key of the transition probability matrix in `adata.obsp` or
        `adata.uns`. Default: "T_forward".
    recompute_matrix : bool, optional
        Whether the matrix should be recomputed. Default: False.
    self_transitions : bool, optional
        Whether the matrix should be computed with self-transitions
        enabled. Default: False.
    num_chains : int, optional
        Number of Markov chains to simulate. Default: 1,000.
    max_iter : int, optional
        Maximum number of iterations for Markov chains. Default: 1,000.
    rtol : float, optional
        Relative tolerance for convergence. Default: 1e-5.
    atol : float, optional
        Absolute tolerance for convergence. Default: 1e-5.
    n_jobs : int, optional
        Number of threads used for Markov sampling when running on CPU.
        Default: -1 (all available threads).
    seed : int, optional
        Random seed. Default: 42.
    copy : bool, optional
        Whether the AnnData object should be copied or modified inplace.
        Default: False (modify inplace).
    """

    adata: ad.AnnData
    """AnnData object for sampling cell sequences and getting gene expression."""
    X: torch.Tensor
    """Gene expression tensor of shape (n_cells, n_genes)."""
    sampler: MarkovSampler
    """Instance of Markov chain sampler for sampling cell sequences."""
    markov_chains: torch.Tensor
    """Markov chains of cell sequences, represented by their index."""
    gene_names: pd.Index
    """Gene names index of shape (n_genes,)."""

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
        X = self.adata.X if layer is None else self.adata.layers[layer]
        if X is None:
            raise ValueError(
                "Invalid gene expression matrix. Make sure that adata.X is present."
            )
        self.gene_names = self.adata.var_names
        if mask_var is not None:
            if isinstance(mask_var, str):
                if adata.var[mask_var].dtype != bool:
                    raise TypeError(
                        f"Expected adata.var[mask_var] to be of type bool, but got type {adata.var[mask_var].dtype}"
                    )
                mask_var = self.adata.var[mask_var].to_numpy()
            X = X[:, mask_var]
            self.gene_names = self.gene_names[mask_var]
        if isinstance(
            X,
            (sparse.csc_array, sparse.csr_array, sparse.csc_matrix, sparse.csr_matrix),
        ):
            X = X.toarray()
        self.X = torch.from_numpy(X)

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

    def n_genes(self):
        return len(self.gene_names)


class PseudotimeDataset(Dataset):
    """
    A pytorch dataset to sample cells along (pseudo) time, ready for
    training an ODE or SDE model with a distributional loss.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    layer : Optional[str], optional
        Layer of the AnnData object to use as the gene expression values.
        If `None`, use adata.X. Default: None.
    mask_var : Optional[Union[NDArray[np.bool_], str]], optional
        Only use a subset of genes given by a boolean array or a string
        referring to a column in `adata.var`. Default: None (use
        all genes).
    pseudotime_key : str, optional
        Key of the (pseudo) time annotation in `adata.obs`. Default:
        "pseudotime".
    num_cells : int, optional
        Number of cells to sample for each time point. Default: 1,000.
    num_quantiles : int, optional
        Number of quantiles for generating for sampling cells along
        (pseudo) time. Default: 20.
    time_points : Optional[Union[NDArray[np.bool_], str]], option
        Explicit time points to use for sampling cells. Overrides
        `num_quantiles`. Default: None (use quantiles along pseudottime).
    seed : int, optional
        Random seed. Default: 42.
    copy : bool, optional
        Whether the AnnData object should be copied or modified inplace.
        Default: False (modify inplace).
    """

    adata: ad.AnnData
    """AnnData object for sampling cell sequences and getting gene expression."""
    X: torch.Tensor
    """Gene expression tensor of shape (n_cells, n_genes)."""
    sampled_cells: torch.Tensor
    """Sampled cell indices of shape (num_cells, len(self.time_points))."""
    gene_names: pd.Index
    """Gene names index of shape (n_genes,)."""

    def __init__(
        self,
        adata: ad.AnnData,
        layer: Optional[str] = None,
        mask_var: Optional[Union[NDArray[np.bool_], str]] = "highly_variable",
        pseudotime_key: str = "pseudotime",
        num_cells: int = 1_000,
        num_quantiles: int = 20,
        time_points: Optional[Sequence[Union[int, float]]] = None,
        seed: int = 42,
        copy: bool = False,
    ):
        if pseudotime_key not in adata.obs:
            raise KeyError("Pseudotime key not found in adata.obs.")
        # Prepare expression data
        self.adata = adata.copy() if copy else adata
        X = self.adata.X if layer is None else self.adata.layers[layer]
        if X is None:
            raise ValueError(
                "Invalid gene expression matrix. Make sure that adata.X is present."
            )
        self.gene_names = self.adata.var_names
        if mask_var is not None:
            if isinstance(mask_var, str):
                if adata.var[mask_var].dtype != bool:
                    raise TypeError(
                        f"Expected adata.var[mask_var] to be of type bool, but got type {adata.var[mask_var].dtype}"
                    )
                mask_var = self.adata.var[mask_var].to_numpy()
            X = X[:, mask_var]
            self.gene_names = self.gene_names[mask_var]
        if isinstance(
            X,
            (sparse.csc_array, sparse.csr_array, sparse.csc_matrix, sparse.csr_matrix),
        ):
            X = X.toarray()
        self.X = torch.from_numpy(X)

        # Sample cells at time points
        pseudotime: pd.Series = adata.obs[pseudotime_key]
        if time_points is None:
            # Use quantile center as bin value
            pseudotime = pd.qcut(pseudotime, num_quantiles).apply(
                lambda x: (x.left + x.right) / 2
            )
            time_points = sorted(pseudotime.cat.categories)
        self.pseudotime = torch.from_numpy(pseudotime.to_numpy())
        self.time_points = torch.from_numpy(np.asarray(time_points))
        self.sampled_cells = torch.zeros(
            num_cells, len(self.time_points), dtype=torch.int
        )
        integer_idx = torch.arange(len(self.adata))
        generator = torch.Generator()
        generator.manual_seed(seed)
        for i, tp in enumerate(time_points):
            tp_idx = integer_idx[self.pseudotime == tp]
            self.sampled_cells[:, i] = tp_idx[
                torch.randint(len(tp_idx), (num_cells,), generator=generator)
            ]

    def __len__(self):
        return self.sampled_cells.shape[0]

    def __getitem__(self, idx):
        return self.X[self.sampled_cells[idx, :], :]

    def n_genes(self):
        return len(self.gene_names)


class VelocityDataset(Dataset):
    """
    A pytorch dataset containing spliced expression and RNA velocity,
    ready for training an ODE model with gradient matching.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    layer : str, optional
        Layer of the AnnData object to use as spliced expression values.
        By default "Ms".
    mask_var : Optional[Union[NDArray[np.bool_], str]], optional
        Only use a subset of genes given by a boolean array or a string
        referring to a column in `adata.var`. By default, None (use
        all genes).
    vkey : str, optional
        Name of velocity estimates to be used. By default "velocity".
    """

    adata: ad.AnnData
    """AnnData object for sampling cell sequences and getting gene expression."""
    X: torch.Tensor
    """Spliced expression tensor of shape (n_cells, n_genes)."""
    V: torch.Tensor
    """RNA velocity tensor of shape (n_cells, n_genes)"""
    gene_names: pd.Index
    """Gene names index of shape (n_genes,)."""

    def __init__(
        self,
        adata: ad.AnnData,
        layer: Optional[str] = "Ms",
        mask_var: Optional[Union[NDArray[np.bool_], str]] = "highly_variable",
        vkey: str = "velocity",
    ):
        # Prepare expression data
        self.adata = adata
        X = self.adata.X if layer is None else self.adata.layers[layer]
        if X is None:
            raise ValueError(
                "Invalid gene expression matrix. Make sure that adata.X is present."
            )
        V = self.adata.layers[vkey]
        self.gene_names = adata.var_names
        if mask_var is not None:
            if isinstance(mask_var, str):
                if adata.var[mask_var].dtype != bool:
                    raise TypeError(
                        f"Expected adata.var[mask_var] to be of type bool, but got type {adata.var[mask_var].dtype}"
                    )
                mask_var = self.adata.var[mask_var].to_numpy()
            X = X[:, mask_var]
            V = V[:, mask_var]
            self.gene_names = self.gene_names[mask_var]
        if isinstance(
            X,
            (sparse.csc_array, sparse.csr_array, sparse.csc_matrix, sparse.csr_matrix),
        ):
            X = X.toarray()
        self.X = torch.from_numpy(X).to(torch.float32)
        self.V = torch.from_numpy(V).to(torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.V[idx, :]

    def n_genes(self):
        return len(self.gene_names)


class BifurcatingMarkovChainDataset(MarkovChainDataset):
    """
    Dummy Markov chain dataset simulating an SDE with a defined
    adjacency matrix instead of using a single-cell dataset.
    """

    def __init__(self, y0: torch.Tensor):
        from bowside.training import integrate_model

        self.sde = BifurcatingSDE()
        n_steps = 50
        self.time_points = torch.linspace(
            0, n_steps - 1, n_steps, dtype=torch.float32, device=y0.device
        )
        simulations = integrate_model(
            y0, self.sde, time_points=self.time_points, dt=0.1
        ).permute((1, 0, 2))
        self.X = simulations.reshape(-1, simulations.shape[-1])
        self.markov_chains = torch.arange(self.X.shape[0]).reshape(
            simulations.shape[0], simulations.shape[1]
        )
        self.gene_names = pd.Index([f"gene_{i}" for i in range(simulations.size(2))])
        self.adata = ad.AnnData(
            X=self.X.detach().cpu().numpy(), var=pd.DataFrame(index=self.gene_names)
        )


class BifurcatingPseudotimeDataset(PseudotimeDataset):
    """
    Dummy pseudotime dataset simulating an SDE with a defined adjacency
    matrix instead of using a single-cell dataset.
    """

    def __init__(self, y0: torch.Tensor):
        from bowside.training import integrate_model

        self.sde = BifurcatingSDE()
        n_steps = 50
        self.time_points = torch.linspace(
            0, n_steps - 1, n_steps, dtype=torch.float32, device=y0.device
        )
        simulations = integrate_model(
            y0, self.sde, time_points=self.time_points, dt=0.1
        ).permute((1, 0, 2))
        self.X = simulations.reshape(-1, simulations.shape[-1])
        self.pseudotime = self.time_points.repeat_interleave(simulations.shape[1])
        self.sampled_cells = torch.arange(self.X.shape[0]).reshape(
            simulations.shape[0], simulations.shape[1]
        )

        self.gene_names = pd.Index([f"gene_{i}" for i in range(simulations.size(2))])
        self.adata = ad.AnnData(
            X=self.X.detach().cpu().numpy(), var=pd.DataFrame(index=self.gene_names)
        )
