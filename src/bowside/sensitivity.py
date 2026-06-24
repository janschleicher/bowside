"""Module for sensitivity analysis on a trained dynamic GRN model."""

from typing import Optional, Tuple, Union

import equinox as eqx
import jax
import numpy as np
import pandas as pd
from jax import numpy as jnp

from bowside.dataset import MarkovChainDataset, PseudotimeDataset
from bowside.model import GRNModel, GRNModelSDE
from bowside.training import LossFunction, integrate_model


def simulate_knockout(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    gene: Union[str, int],
    loss_func: LossFunction,
    key: Optional[jax.Array] = None,
    loss_kwargs: dict = {},
) -> Tuple[jax.Array, jax.Array, float]:
    """
    Simulate gene expression sequences with a single gene knockout by
    setting the weights of all edges outgoing from the respective node
    in the GRN model to 0 and compute the global effect of this knockout.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for knockout simulations (starting cells and time points).
    model : GRNModel
        The trained gene regulatory network model.
    gene : Union[str, int]
        The gene to knock out. Either an integer index or the gene name.
    loss_func : LossFunction
        Loss function to use for quantifying the effect of the knockout.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
    solver_method : Optional[str], optional
        Numerical solver to use. With the default None, a suitable method
        is chosen depending on the model type.
    dt : float, optional
        The step size for the solver. Only used for SDEs. Default: 0.001.
    atol : float, optional
        Absolute tolerance. Default: 1e-4.
    rtol : float, optional
        Relative tolerance. Default: 1e-5.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    Tuple[jax.Array, jax.Array, float]
        Simulations for the wildtype and knockout model, and loss
        function value between knockout and wild type simulations as a
        measure of the global effect of the knockout of this gene.

    Also sets the following fields in `dataset.adata.uns`:

    `adata.uns["bowside"]["simulations"]` : :class:`numpy.ndarray`
        Array of shape (n_simulations, n_steps, n_var) containing gene
        expression simulations from the trained model.
    `adata.uns["bowside"]["simulations_ko_{gene}"]` : :class:`numpy.ndarray`
        Array of shape (n_simulations, n_steps, n_var) containing gene
        expression simulations from the trained model, with a knockout
        of `gene`.

    Raises
    ------
    TypeError
        An error is raised if `dataset` or `gene` are of the wrong type.
    ValueError
        An error is raised if the number of genes in the dataset and in
        the model do not match.
    KeyError
        An error is raised if `gene` is of type str and is not a valid
        gene name in `dataset`.
    IndexError
        An error is raised if `gene` is of type i
        nt and is out of bounds.
    """
    device = model.adj.device
    time_points = None
    if isinstance(dataset, PseudotimeDataset):
        time_points = jnp.asarray(dataset.time_points.numpy().copy(), device=device)
    elif isinstance(dataset, MarkovChainDataset):
        n_steps = dataset.markov_chains.shape[1]
        time_points = jnp.linspace(0, n_steps - 1, n_steps, device=device)
    else:
        raise TypeError(
            f"Invalid dataset type: {type(dataset)}. Must be one of "
            f"[MarkovChainDataset, PseudotimeDataset]."
        )

    # Data setup
    y = jnp.asarray(dataset[:].numpy().copy(), device=device)
    y0 = y[:, 0, :]

    # Check parameters
    if dataset.n_genes() != model.n_genes():
        raise ValueError(
            f"Incompatible number of genes. Dataset: {dataset.n_genes()}, model: {model.n_genes()}"
        )

    # Find the gene for the KO
    if isinstance(gene, str):
        if gene not in dataset.gene_names:
            raise KeyError(f"Gene {gene} not found in the dataset.")
        gene_idx = dataset.gene_names.get_loc(gene)
    elif isinstance(gene, int):
        if gene < 0 or gene >= dataset.n_genes():
            raise IndexError(
                f"Gene index {gene} out of bounds for dataset with "
                f"{dataset.n_genes()} genes."
            )
        gene_idx = gene
    else:
        raise TypeError(
            f"Parameter `gene` must be of type str or int, but was {type(gene)}"
        )
    assert isinstance(gene_idx, int)

    # Remove all outgoing edges from the knockout gene
    adj_mask = model.adj_mask.at[gene_idx, :].set(0)
    model_ko = eqx.tree_at(lambda m: m.adj_mask, model, adj_mask)

    # Get wildtype simulations
    if "bowside" not in dataset.adata.uns:
        dataset.adata.uns["bowside"] = {}
    if "simulations" in dataset.adata.uns["bowside"]:
        sequences_wt = jnp.asarray(dataset.adata.uns["bowside"]["simulations"])
    else:
        if isinstance(model, GRNModelSDE):
            assert key is not None, "For a `GRNModelSDE`, a random key is required."
            key, sub_key = jax.random.split(key)
        else:
            sub_key = None
        sequences_wt = integrate_model(model, y0, time_points, key=sub_key)
        dataset.adata.uns["bowside"]["simulations"] = np.asarray(sequences_wt)

    # Perform knockout simulations
    if isinstance(model, GRNModelSDE):
        assert key is not None, "For a `GRNModelSDE`, a random key is required."
        key, sub_key = jax.random.split(key)
    else:
        sub_key = None
    y0_ko = y0.at[gene_idx].set(0)
    sequences_ko = integrate_model(model_ko, y0_ko, time_points, sub_key)

    # Set knockout gene expression to 0
    sequences_ko = sequences_ko.at[:, :, gene_idx].set(0)

    dataset.adata.uns["bowside"][f"simulations_ko_{dataset.gene_names[gene_idx]}"] = (
        jnp.asarray(sequences_ko)
    )

    # Compute difference between wildtype and knockout, excluding the knockout gene
    gene_selection = jnp.arange(dataset.n_genes()) != gene_idx
    diff = loss_func(
        sequences_ko[:, :, gene_selection],
        sequences_wt[:, :, gene_selection],
        **loss_kwargs,
    ).item()

    return sequences_wt, sequences_ko, diff


def simulate_all_knockouts(
    dataset: Union[MarkovChainDataset, PseudotimeDataset],
    model: GRNModel,
    loss_func: LossFunction,
    key: Optional[jax.Array],
    loss_kwargs: dict = {},
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Simulate gene expression sequences for single-gene knockouts of each
    gene in `dataset` by setting the weights of all edges outgoing from
    the respective node in the GRN model to 0 and compute the global and
    edge-specific effects of all knockouts.

    Parameters
    ----------
    dataset : Union[MarkovChainDataset, PseudotimeDataset]
        Dataset to use for knockout simulations (starting cells and time points).
    model : GRNModel
        The trained gene regulatory network model.
    loss_func : LossFunction
        Loss function to use for quantifying the effect of the knockout.
    key : Optional[jax.Array], optional
        Key for the random number generator. Required for integration
        with a GRNModelSDE. Default: None.
    loss_kwargs : dict, optional
        Additional keyword arguments for the loss function. Default: {}.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Global and edge-specific effects of each knockout.

    Also sets the following fields in `dataset.adata.uns`:

    `adata.uns["bowside"]["simulations"]` : :class:`numpy.ndarray`
        Array of shape (n_simulations, n_steps, n_var) containing gene
        expression simulations from the trained model.
    `adata.uns["bowside"]["simulations_ko_{gene}"]` : :class:`numpy.ndarray`
        Array of shape (n_simulations, n_steps, n_var) containing gene
        expression simulations from the trained model for single-gene
        knockouts of all genes in `dataset`.
    `adata.uns["bowside"]["sensitivity"]["global_knockout_effects"] : :class:`pandas.Series`
        Series of length n_var containing the loss function values between
        knockout and wild type simulations as a measure of the global
        effect of the single-gene knockouts of all genes in `dataset`.
    `adata.uns["bowside"]["sensitivity"]["edge_specific_knockout_effects"] : :class:`pandas.DataFrame`
        DataFrame of shape (n_var, n_var) containing the gene-wise loss
        function values between knockout and wild type simulations as a
        measure of the edge-speficic effects of the single-gene knockouts
        of all genes in `dataset`.
    """
    global_diffs = {}
    effects = np.zeros((dataset.n_genes(), dataset.n_genes()))

    # Simulate knockouts for all genes in the dataset
    if isinstance(model, GRNModelSDE):
        assert key is not None, "For a `GRNModelSDE`, a random key is required."
        keys = jax.random.split(key, dataset.n_genes())
    else:
        keys = [None] * dataset.n_genes()
    for i in range(dataset.n_genes()):
        sequences_wt, sequences_ko, diff = simulate_knockout(
            dataset,
            model,
            i,
            loss_func,
            key=keys[i],
            loss_kwargs=loss_kwargs,
        )
        gene = dataset.gene_names[i]
        global_diffs[gene] = diff

        # Compute knockout effect on each gene separately
        for j in range(dataset.n_genes()):
            if j == i:
                effects[i, j] = np.nan
                continue
            effects[i, j] = loss_func(
                jnp.expand_dims(sequences_ko[:, :, j], 2),
                jnp.expand_dims(sequences_wt[:, :, j], 2),
                **loss_kwargs,
            ).item()

    # Save results
    global_diffs = pd.Series(global_diffs, name="difference")
    effects = pd.DataFrame(
        data=effects, index=dataset.gene_names, columns=dataset.gene_names
    )
    if "sensitivity" not in dataset.adata.uns["bowside"]:
        dataset.adata.uns["bowside"]["sensitivity"] = {}
    dataset.adata.uns["bowside"]["sensitivity"]["global_knockout_effects"] = (
        global_diffs
    )
    dataset.adata.uns["bowside"]["sensitivity"]["edge_specific_knockout_effects"] = (
        effects
    )

    return global_diffs, effects
