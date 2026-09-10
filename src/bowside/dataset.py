"""Module for datasets to be used to train an ODE GRN model."""

from collections.abc import Sequence
from math import floor
from typing import Optional, Tuple, Union

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
        referring to a column in `adata.var`. Default: "highly_variable".
        When set to None, use all genes.
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

    def __len__(self) -> int:
        return self.markov_chains.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return self.X[self.markov_chains[idx, :], :]

    def n_genes(self) -> int:
        """
        Get the number of genes in the dataset.

        Returns
        -------
        int
            Number of genes.
        """
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
        referring to a column in `adata.var`. Default: "highly_variable".
        When set to None, use all genes.
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
    pseudotime: torch.Tensor
    """Tensor of shape (n_cells,) with pseudotime of each cell in `adata`."""
    time_points: torch.Tensor
    """Tensor of shape (num_quantiles,) with the quantile means."""

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

    def __len__(self) -> int:
        return self.sampled_cells.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return self.X[self.sampled_cells[idx, :], :]

    def n_genes(self) -> int:
        """
        Get the number of genes in the dataset.

        Returns
        -------
        int
            Number of genes.
        """
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
        referring to a column in `adata.var`. Default: "highly_variable".
        When set to None, use all genes.
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

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx, :], self.V[idx, :]

    def n_genes(self) -> int:
        """
        Get the number of genes in the dataset.

        Returns
        -------
        int
            Number of genes.
        """
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
            self.sde, y0, time_points=self.time_points, dt=0.1
        )
        self.X = simulations.reshape(-1, simulations.shape[-1])
        self.markov_chains = (
            torch.arange(self.X.shape[0])
            .reshape(simulations.shape[0], simulations.shape[1])
            .permute(1, 0)
            .contiguous()
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
            self.sde, y0, time_points=self.time_points, dt=0.1
        )
        self.X = simulations.reshape(-1, simulations.shape[-1])
        self.pseudotime = self.time_points.repeat_interleave(simulations.shape[1])
        self.sampled_cells = (
            torch.arange(self.X.shape[0])
            .reshape(simulations.shape[0], simulations.shape[1])
            .permute(1, 0)
            .contiguous()
        )

        self.gene_names = pd.Index([f"gene_{i}" for i in range(simulations.size(2))])
        self.adata = ad.AnnData(
            X=self.X.detach().cpu().numpy(), var=pd.DataFrame(index=self.gene_names)
        )


def split_by_time_point(
    dataset: PseudotimeDataset,
    train_proportion: float = 0.8,
    validation_proportion: float = 0.2,
    test_proportion: float = 0.0,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Split the dataset by time points. The validation time points are
    taken from the end, the test time points from the middle of the
    remaining time points. By default, only training and validation
    indices are generated.

    Parameters
    ----------
    dataset : PseudotimeDataset
        Dataset that should be split.
    train_proportion : float, optional
        Proportion of timepoints for training. Default: 0.8.
    validation_proportion : float, optional
        Proportion of timepoints for validation. Default: 0.2.
    test_proportion : float, optional
        Proportion of timepoints for testing. Default: 0.0.

    Returns
    -------
    Union[ Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor] ]
        Indices of the training, validation, and (optionally) test
        timepoints.

    Raises
    ------
    ValueError
        An error is raised if one of the proportions is not in the
        appropriate range, or if they do not sum to 1.
    """
    if not np.isclose(train_proportion + validation_proportion + test_proportion, 1.0):
        raise ValueError("Proportions must sum up to 1.")

    if train_proportion <= 0.0 or train_proportion >= 1.0:
        raise ValueError(
            f"Training proportion must be in (0, 1), but was {train_proportion}."
        )
    if validation_proportion <= 0.0 or validation_proportion >= 1.0:
        raise ValueError(
            f"Validation proportion must be in (0, 1), but was {validation_proportion}."
        )
    if test_proportion < 0.0 or test_proportion >= 1.0:
        raise ValueError(
            f"Test proportion must be in [0, 1), but was {test_proportion}."
        )

    n_time_points = len(dataset.time_points)
    all_indices = torch.arange(n_time_points)

    # Take validation time points from the end
    n_validation = floor(validation_proportion * n_time_points)
    validation_indices = all_indices[-n_validation:]

    train_indices = all_indices[:-n_validation]

    # To test model performance based on interpolation, take time points
    # from the middle of the remaining time points
    if test_proportion > 0:
        n_test = floor(test_proportion * n_time_points)
        test_start_idx = ((n_time_points - n_validation) // 2) - (n_test // 2)
        test_indices = all_indices[test_start_idx : test_start_idx + n_test]
        train_indices = torch.tensor(
            [idx for idx in train_indices if idx not in test_indices]
        )
        return train_indices, validation_indices, test_indices
    return train_indices, validation_indices


class PseudotimeTimepointWrapper(Dataset):
    """
    Wrapper around the :class:`PseudotimeDataset` to select a subset of
    the original time points, e.g., for train/validation/test split.

    Parameters
    ----------
    dataset : PseudotimeDataset
        Original dataset.
    timepoint_indices : torch.Tensor
        Subset of time points.
    """

    def __init__(self, dataset: PseudotimeDataset, timepoint_indices: torch.Tensor):
        self.dataset = dataset
        self.time_point_indices = timepoint_indices
        self.time_points = dataset.time_points[self.time_point_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if sample.dim() == 2:
            sliced_sample = sample[self.time_point_indices, :]
        elif sample.dim() == 3:
            sliced_sample = sample[:, self.time_point_indices, :]
        else:
            raise ValueError(
                f"Unexpected tensor shape from underlying dataset: {sample.shape}"
            )
        return sliced_sample

    def n_genes(self) -> int:
        """
        Get the number of genes in the dataset.

        Returns
        -------
        int
            Number of genes.
        """
        return self.dataset.n_genes()
