"""Module for Markov chain sampling."""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import torch
from tqdm.auto import tqdm

from .utils import check_transition_matrix

log = logging.getLogger(__name__)


class MarkovSampler:
    """
    A class for sampling Markov chains of cells from an AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object for sampling cell sequences.
    tpm_key : str, optional
        Key of the transition probability matrix in `adata.obsp` or
        `adata.uns`, by default "T_forward".
    recompute_matrix : bool, optional
        Whether the matrix should be recomputed, by default False.
    self_transitions : bool, optional
        Whether the matrix should be computed with self-transitions
        enabled, by default False.
    copy : bool, optional
        Whether the AnnData object should be copied or modified inplace,
        by default False (modify inplace).
    """

    adata: ad.AnnData
    """AnnData object for sampling cell sequences."""
    copy: bool
    """Whether the AnnData object is a copy of the input."""
    tpm: torch.Tensor
    """Transition probability matrix describing the Markov chain of shape (n_cells, n_cells)."""
    p_init: torch.Tensor
    """Initial state probability of shape (n_cells)."""
    p_term: torch.Tensor
    """Stationary distribution of shape (n_cells)."""
    convergence: int
    """Number of steps to convergence to stationary distribution."""

    def __init__(
        self,
        adata: ad.AnnData,
        tpm_key: str = "T_forward",
        recompute_matrix: bool = False,
        self_transitions: bool = False,
        copy: bool = False,
    ) -> None:
        self.copy = copy
        self.adata = adata.copy() if self.copy else adata
        check_transition_matrix(
            self.adata,
            tpm_key=tpm_key,
            recompute_matrix=recompute_matrix,
            self_transitions=self_transitions,
        )
        indptr = torch.from_numpy(self.adata.obsp[tpm_key].indptr).to(torch.int32)
        indices = torch.from_numpy(self.adata.obsp[tpm_key].indices).to(torch.int32)
        data = torch.from_numpy(self.adata.obsp[tpm_key].data)
        shape = self.adata.obsp[tpm_key].shape
        self.tpm = torch.sparse_csr_tensor(
            indptr, indices, data, size=shape, dtype=torch.float64
        )
        self.p_init = torch.tensor(
            self.adata.obs["root_cells"].values, dtype=torch.float64
        )
        self.p_init = self.p_init / self.p_init.sum()

    def sample_markov_chains(
        self,
        num_chains: int = 1_000,
        max_iter: int = 1_000,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> Optional[ad.AnnData]:
        """
        Sample Markov chains of cells.

        Parameters
        ----------
        num_chains : int, optional
            Number of Markov chains to simulate, by default 1,000.
        max_iter : int, optional
            Maximum number of iterations for Markov chains, by default 1,000.
        rtol : float, optional
            Relative tolerance for convergence, by default 1e-5.
        atol : float, optional
            Absolute tolerance for convergence, by default 1e-5.
        n_jobs : int, optional
            Number of threads when running on CPU, by default -1 (all
            available threads).
        seed : int, optional
            Random seed, by default 42.

        Returns
        -------
        Optional[ad.AnnData]
            Returns `None` if `self.copy=False`, else returns an :class:`AnnData` object.
            Sets the following field:

            `adata.uns["markov_sampling"]`: :class:`dict`, with "cell_sequences"
                (:class:`numpy.ndarray`) and "cell_sequences_max_iter"
                (:class:`numpy.ndarray`), containing the indices of sampled cells up to
                convergence and `self.max_iter`, respectively.
        """
        self._compute_stationary_distribution(max_iter=max_iter, rtol=rtol, atol=atol)
        markov_chains_max_iter = self._iterate_markov_chains(
            num_chains=num_chains, max_iter=max_iter, n_jobs=n_jobs, seed=seed
        )

        # Store in AnnData object as numpy arrays
        res = {
            "cell_sequences": markov_chains_max_iter[:, : self.convergence]
            .detach()
            .numpy(),
            "cell_sequences_max_iter": markov_chains_max_iter.detach().numpy(),
            "convergence": self.convergence,
            "params": {
                "num_chains": num_chains,
                "max_iter": max_iter,
                "atol": atol,
                "rtol": rtol,
            },
        }
        self.adata.uns["markov_sampling"] = res

        if self.copy:
            return self.adata

    def _compute_stationary_distribution(
        self, max_iter: int = 1_000, rtol: float = 1e-5, atol: float = 1e-5
    ) -> None:
        """
        Compute the stationary distribution of the Markov chain and the
        number of steps to convergence.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations for Markov chains, by default 1,000.
        rtol : float, optional
            Relative tolerance for convergence, by default 1e-5.
        atol : float, optional
            Absolute tolerance for convergence, by default 1e-5.
        """
        converged = False
        pi = self.p_init

        # Iterative multiplication with transition probability matrix
        for i in range(max_iter):
            pi_ = pi @ self.tpm
            if torch.allclose(pi_, pi, rtol=rtol, atol=atol):
                converged = True
                break
            else:
                pi = pi_

        if converged:
            log.info(
                f"Convergence to stationary distribution reached after {i + 1} steps."
            )
        else:
            log.warning(
                f"The Markov chain did not converge to the stationary distribution "
                f"after {max_iter} steps"
            )

        self.convergence = i + 1
        self.p_term = pi

    def _iterate_markov_chains(
        self,
        num_chains: int = 1_000,
        max_iter: int = 1_000,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Sample Markov chains of cells from the transition probability
        matrix.

        Parameters
        ----------
        num_chains : int, optional
            Number of Markov chains to simulate, by default 1,000.
        max_iter : int, optional
            Maximum number of iterations for Markov chains, by default 1,000.
        n_jobs : int, optional
            Number of threads when running on CPU, by default -1 (all
            available threads).
        seed : int, optional
            Random seed, by default 42.

        Returns
        -------
        torch.Tensor
            Cell sequences of shape (num_chains, max_iter).
        """
        # Select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            if n_jobs < 0:
                n_jobs = torch.get_num_threads() + 1 + n_jobs
            torch.set_num_threads(n_jobs)
        torch.manual_seed(seed)
        np_rng = np.random.default_rng(seed)

        # Move tensors to device
        tpm = self.tpm.to(device)
        p_init = self.p_init.detach().numpy()

        # Precompute cumsum for each row
        tpm_ = tpm.to_dense().cumsum(1)

        # Parallel random steps and initial states
        random_steps = torch.rand(num_chains, max_iter - 1, device=device)
        state_indices = torch.empty(
            num_chains, max_iter, dtype=torch.int32, device=device
        )

        # Initial states: sample for each chain
        state_indices[:, 0] = torch.from_numpy(
            np_rng.choice(tpm.shape[0], size=num_chains, replace=True, p=p_init)
        ).to(device)

        # Sample cell sequences
        for step in tqdm(
            range(1, max_iter),
            desc="Sampling cell sequences",
            initial=1,
            total=max_iter,
            unit="step(s)",
        ):
            prev_states = state_indices[:, step - 1]
            probs = tpm_[prev_states]
            state_indices[:, step] = torch.searchsorted(
                probs, random_steps[:, step - 1].unsqueeze(1), side="right"
            ).squeeze(1)

        return state_indices.detach()
