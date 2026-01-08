import logging

import anndata as ad
import numpy as np
import scvelo as scv

log = logging.getLogger(__name__)


def check_transition_matrix(
    adata: ad.AnnData,
    tpm_key: str = "T_forward",
    recompute_matrix: bool = False,
    self_transitions: bool = False,
):
    """
    Check if a proper transition probability matrix is present.
    Compute the matrix if it is not present or if it should be
    recomputed.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object.
    tpm_key : str, optional
        Key of the transition probability matrix in `adata.obsp` or
        `adata.uns`, by default "T_forward".
    recompute_matrix : bool, optional
        Whether the matrix should be recomputed, by default False.
    self_transitions : bool, optional
        Whether the matrix should be computed with self-transitions
        enabled, by default False.
    """
    compute_matrix = False

    # Check if transition probability matrix has to be computed
    if recompute_matrix:
        compute_matrix = True
    if tpm_key not in adata.obsp.keys():
        if tpm_key in adata.uns:
            adata.obsp[tpm_key] = adata.uns[tpm_key]
        else:
            log.warning(
                "Transition probability matrix was not present in .obsp "
                "or .uns and is computed from RNA velocity."
            )
            compute_matrix = True

    compute_terminal_states = (
        compute_matrix
        or ("root_cells" not in adata.obs.columns)
        or ("end_points" not in adata.obs.columns)
    )

    # Compute matrix and terminal states if required
    if compute_matrix:
        adata.obsp[tpm_key] = scv.utils.get_transition_matrix(
            adata=adata, self_transitions=self_transitions
        ).toarray()

    if compute_terminal_states:
        scv.tl.terminal_states(adata, self_transitions=self_transitions)

    # Check row sums
    assert np.allclose(adata.obsp[tpm_key].sum(axis=1), 1.0), (
        "The rows of the transition probability matrix do not sum to 1."
    )
