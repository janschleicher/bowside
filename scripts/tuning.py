import argparse
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import networkx as nx
import optuna
import pandas as pd
import scanpy as sc
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from bowside.dataset import PseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.training import (
    NaNTensorError,
    qqloss,
    sinkhorn_loss,
    train_integration,
)


def objective(
    trial: optuna.Trial,
    input_path: str,
    out_dir: Path,
    layer: str,
    mask_var: str,
    pseudotime_key: str,
    num_cells: int,
    num_quantiles: int,
    time_points: Optional[Sequence[Union[int, float]]],
    adj_list: Optional[pd.DataFrame],
    dt: float,
    sigma_min: float,
    sigma_max: float,
    loss_func: str,
    weighted_loss: bool,
    regularization: Optional[Literal["l1", "l2"]],
    regularization_strength: Optional[float],
    regularization_strength_min: float,
    regularization_strength_max: float,
    epochs_min: int,
    epochs_max: int,
    train_proportion: float,
    test_proportion: float,
    time_point_split: bool,
    patience: int,
) -> float:
    """
    Optuna objective function.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial.
    input_path : str
        Path to input AnnData object.
    out_dir : Path
        Output directory.
    layer : str
        Layer of the AnnData object to use as the gene expression values.
    mask_var : str
        Only use a subset of genes given by a boolean array or a string
        referring to a column in `adata.var`.
    pseudotime_key : str
        Key of the (pseudo) time annotation in `adata.obs`.
    num_cells : int
        Number of cells to sample for each time point.
    num_quantiles : int
        Number of quantiles for generating for sampling cells along
        (pseudo) time.
    time_points : Optional[Sequence[Union[int, float]]]
         Explicit time points to use for sampling cells. Overrides
        `num_quantiles`. When set to None, quantiles along pseudottime
        are used.
    adj_list : Optional[pd.DataFrame]
        Adjacency list for model adjacency matrix intialization. When
        set to None, a fully connected matrix is used.
    dt : float
        The step size for the solver.
    sigma_min : float
        Minimum value of the level of the (diagonal) noise.
    sigma_max : float
        Maximum value of the level of the (diagonal) noise.
    loss_func : str
        Loss function to optimize.
    weighted_loss : bool
        Whether to use a loss with higher weights for later time points.
    regularization : Optional[Literal["l1", "l2"]]
        Regularization method for the adjacency method. Options are "l1"
        and "l2". When set to None, no regularization is applied.
    regularization_strength : Optional[float]
        Relative weight of the regularization term added to the loss if
        `regularization` is not None. When set to None, the strength
        of regularization is tuned alongside other hyperparameters.
    regularization_strength_min : float
        Minimum regularization strength when it is tuned.
    regularization_strength_max : float
        Maximum regularization strength when it is tuned.
    epochs_min : int
        Minimum number of training epochs.
    epochs_max : int
        Maximum number of training epochs.
    train_proportion : float
        Proportion of samples to use for training.
    test_proportion : float
        Proportion of time points to use for testing of interpolation
        performance. Only applicable if `time_point_split` is True. The
        evaluation on the test time points is not done inside this
        hyperparameter optimization.
    time_point_split : bool
        Whether to split into training, validation, and optionally test
        data by time point instead of by sample.
    patience : int
        Number of epochs to wait for an improvement before stopping.

    Returns
    -------
    float
        Value of the final validation loss.

    Raises
    ------
    optuna.TrialPruned
        Raises an error if incompatible hyperparameter values are chosen,
        if numerical instabilities arise during integration, or if the
        loss cannot be computed.
    """
    # Set seed for this trial
    torch.manual_seed(trial.number)

    # Setup per-trial logger
    checkpoint_dir = out_dir / str(trial.number)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    # Attach handler to bowside.training logger
    logger = logging.getLogger(f"optuna_trial_{trial.number}")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info(f"Logger setup complete for trial {trial.number}")

    # Data loading
    adata = sc.read_h5ad(input_path)
    dataset = PseudotimeDataset(
        adata=adata,
        layer=layer,
        mask_var=mask_var,
        pseudotime_key=pseudotime_key,
        num_cells=num_cells,
        num_quantiles=num_quantiles,
        time_points=time_points,
    )

    # Model parameters (encode booleans as strings for Optuna Dashboard)
    adj_init_normal = (
        trial.suggest_categorical("adj_init_normal", ["True", "False"]) == "True"
    )
    sigma = trial.suggest_float("sigma", sigma_min, sigma_max, log=True)
    shared_sigma = (
        trial.suggest_categorical("shared_sigma", ["True", "False"]) == "True"
    )
    fixed_sigma = trial.suggest_categorical("fixed_sigma", ["True", "False"]) == "True"
    constant_noise = (
        trial.suggest_categorical("constant_noise", ["True", "False"]) == "True"
    )

    if not shared_sigma and fixed_sigma:
        raise optuna.TrialPruned("Cannot use fixed, gene-wise sigma.")

    # Model setup
    adj = None
    if adj_list is not None:
        adj_list = adj_list[
            (adj_list.source.isin(dataset.gene_names))
            & (adj_list.target.isin(dataset.gene_names))
        ]
        grn = nx.from_pandas_edgelist(
            adj_list, source="source", target="target", create_using=nx.DiGraph
        )
        for gene in dataset.gene_names:
            if gene not in grn.nodes():
                grn.add_node(gene)
        adj = torch.from_numpy(nx.to_numpy_array(grn, nodelist=dataset.gene_names)).to(
            torch.float32
        )

    model = GRNModelSDE(
        dataset.n_genes(),
        adj=adj,
        adj_init_normal=adj_init_normal,
        sigma=sigma,
        shared_sigma=shared_sigma,
        fixed_sigma=fixed_sigma,
        constant_noise=constant_noise,
    )

    # Optimizer setup
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    # Training parameters
    num_epochs = trial.suggest_int(
        "num_epochs", low=epochs_min, high=epochs_max, step=500
    )
    solver_method = trial.suggest_categorical("solver_method", ["euler", "milstein"])

    # Regularization
    if regularization is not None:
        if regularization_strength is None:
            regularization_strength = trial.suggest_float(
                "regularization_strength",
                regularization_strength_min,
                regularization_strength_max,
                log=True,
            )
    else:
        regularization_strength = (
            0.0001  # not used, but necessary for calling the function
        )

    # Training
    weight = None
    if weighted_loss:
        weight = torch.sqrt(torch.linspace(0.0, 10.0**2, dataset.time_points.size(0)))
    if loss_func == "sinkhorn":
        loss_fn = sinkhorn_loss
    elif loss_func == "qqloss":
        loss_fn = partial(
            qqloss, metric=torch.nn.functional.mse_loss, dim=1, n_quantiles=100
        )

    try:
        _, val_losses = train_integration(
            dataset,
            model,
            optimizer,
            loss_fn,
            loss_weights=weight,
            regularization=regularization,
            regularization_strength=regularization_strength,
            train_proportion=train_proportion,
            test_proportion=test_proportion,
            time_point_split=time_point_split,
            num_epochs=num_epochs,
            patience=patience,
            solver_method=solver_method,
            dt=dt,
            atol=1e-9,
            checkpoint="final",
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )
    except NaNTensorError:
        raise optuna.TrialPruned("Numerical instabilities during SDE integration.")
    except ValueError as e:
        if str(e) == "Maximum allowed size exceeded":
            raise optuna.TrialPruned(
                "Sinkhorn loss cannot be computed (scaling issues)."
            )
        else:
            raise e

    logger.info(f"Final validation loss: {val_losses[-1]}")
    file_handler.flush()
    logger.removeHandler(file_handler)
    file_handler.close()
    return val_losses[-1]


def run_optimization(
    objective_func: Callable,
    out_dir: Path,
    study_name: str,
    dataset: str,
    num_cells: int,
    num_quantiles: int,
    loss_func: str,
    weighted_loss: bool,
    regularization: Optional[Literal["l1", "l2"]],
    regularization_strength: Optional[float],
    train_proportion: float,
    test_proportion: float,
    time_point_split: bool,
    n_trials: int,
):
    """
    Run an optuna hyperparameter optimization.

    Parameters
    ----------
    objective_func : Callable
        Optuna objective function.
    out_dir : Path
        Output directory.
    study_name : str
        Name of the optuna study.
    dataset : str
        Path to input AnnData object.
    num_cells : int
        Number of cells to sample for each time point.
    num_quantiles : int
        Number of quantiles for generating for sampling cells along
        (pseudo) time.
    loss_func : str
        Loss function to optimize.
    weighted_loss : bool
        Whether to use a loss with higher weights for later time points.
    regularization : Optional[Literal["l1", "l2"]]
        Regularization method for the adjacency method. Options are "l1"
        and "l2". When set to None, no regularization is applied.
    regularization_strength : Optional[float]
        Relative weight of the regularization term added to the loss if
        `regularization` is not None. When set to None, the strength
        of regularization is tuned alongside other hyperparameters.
    train_proportion : float
        Proportion of samples to use for training.
    test_proportion : float
        Proportion of time points to use for testing of interpolation
        performance. Only applicable if `time_point_split` is True. The
        evaluation on the test time points is not done inside this
        hyperparameter optimization.
    time_point_split : bool
        Whether to split into training, validation, and optionally test
        data by time point instead of by sample.
    n_trials : int
        Number of optuna trials to run.
    """
    study = optuna.create_study(
        storage=JournalStorage(
            JournalFileBackend(file_path=str(out_dir / "journal.log"))
        ),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("dataset", dataset)
    study.set_user_attr("num_cells", num_cells)
    study.set_user_attr("num_quantiles", num_quantiles)
    study.set_user_attr("loss_function", loss_func)
    study.set_user_attr("weighted_loss", str(weighted_loss))
    study.set_user_attr("regularization", regularization)
    study.set_user_attr("regularization_strength", regularization_strength)
    study.set_user_attr("train_proportion", train_proportion)
    study.set_user_attr(
        "validation_proportion", 1.0 - train_proportion - test_proportion
    )
    study.set_user_attr("test_proportion", test_proportion)
    study.set_user_attr("time_point_split", str(time_point_split))
    study.optimize(objective_func, n_trials=n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter tuning")
    parser.add_argument("-i", "--input", type=str, help="Path to input AnnData object.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output directory. Will be created if it does not exist.",
    )
    parser.add_argument("-n", "---name", type=str, help="Study name.")
    parser.add_argument(
        "--layer", type=str, help="Layer in adata to use for expression values."
    )
    parser.add_argument(
        "--mask-var", type=str, help="Boolean column in adata.var to filter genes."
    )
    parser.add_argument(
        "--pseudotime-key",
        type=str,
        default="pseudotime",
        help="Key of the pseudotime in adata.obs.",
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=1000,
        help="Number of cells per (pseudo)time point.",
    )
    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=20,
        help="Number of quantiles forgenerating for sampling cells along (pseudo) time.",
    )
    parser.add_argument(
        "--time-points",
        type=float,
        nargs="+",
        default=None,
        help="Explicit time points to use for dataset construction.",
    )
    parser.add_argument(
        "--adj-file",
        type=str,
        help="CSV file containg the adjacency list for initialization.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--adj-source-col",
        type=str,
        default="source",
        help="Column containing the source nodes.",
    )
    parser.add_argument(
        "--adj-target-col",
        type=str,
        default="target",
        help="Column containing the target nodes.",
    )
    parser.add_argument(
        "--adj-filter-col",
        type=str,
        help="Column for filtering the adjacency list.",
        default=None,
    )
    parser.add_argument(
        "--adj-filter-value",
        type=str,
        default=None,
        help="Value to use for filtering the adjacency matrix."
        "If this is not set, but the adjacency matrix is filtered, "
        "it will assume boolean values in the filtered column.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="The step size for the solver. This should be chosen "
        "small enough for good numeric integration, but high enough "
        "w.r.t. the time scale of the dataset to reduce compute time.",
    )
    parser.add_argument(
        "--sigma-min", type=float, default=1e-2, help="Minimum value for sigma."
    )
    parser.add_argument(
        "--sigma-max", type=float, default=10, help="Maximum value for sigma."
    )
    parser.add_argument(
        "--loss-func",
        type=str,
        default="sinkhorn",
        choices=["sinkhorn", "qqloss"],
        help="Loss function to use.",
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use a weighted average for the loss function, increasing "
        "the importance of later time points.",
    )
    parser.add_argument(
        "--train-proportion",
        type=float,
        default=0.8,
        help="Proportion of samples/time points to use for training.",
    )
    parser.add_argument(
        "--test-proportion",
        type=float,
        default=0.0,
        help="Proportion of time points to use for interpolation testing. "
        "This can only be used when splitting by time point, and the "
        "corresponding evaluation must be done externally.",
    )
    parser.add_argument(
        "--time-point-split",
        action="store_true",
        help="Split the dataset into train, validation, and optionally "
        "test by time point.",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        choices=["l1", "l2"],
        required=False,
        default=None,
        help="Regularization of adjacency matrix weights.",
    )
    parser.add_argument(
        "--regularization-strength",
        type=float,
        required=False,
        default=None,
        help="Adjacency matrix regularization strength. With the "
        "default `None`, treated as hyperparameter.",
    )
    parser.add_argument(
        "--regularization-strength-min",
        type=float,
        required=False,
        default=1e-5,
        help="Minimum value for the adjacency matrix regularization "
        "strength when treated as a tunable hyperparameter.",
    )
    parser.add_argument(
        "--regularization-strength-max",
        type=float,
        required=False,
        default=1e-2,
        help="Minimum value for the adjacency matrix regularization "
        "strength when treated as a tunable hyperparameter.",
    )
    parser.add_argument(
        "--epochs-min",
        type=int,
        required=False,
        default=500,
        help="Minimum number of epochs.",
    )
    parser.add_argument(
        "--epochs-max",
        type=int,
        required=False,
        default=5000,
        help="Maximum number of epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=100,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials for optuna tuning."
    )
    args = parser.parse_args()

    # Load and filter adjacency list
    adj_list = None
    if args.adj_file is not None:
        adj_list = pd.read_csv(args.adj_file)
        if args.adj_filter_col is not None:
            if args.adj_filter_value is None:
                adj_list = adj_list[adj_list[args.adj_filter_col]]  # Assume boolean
            else:
                adj_list = adj_list[
                    adj_list[args.adj_filter_col] == args.adj_filter_value
                ]
        adj_list = adj_list.rename(
            columns={args.adj_source_col: "source", args.adj_target_col: "target"}
        )
        adj_list = adj_list[["source", "target"]]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup objective
    objective_func = partial(
        objective,
        input_path=args.input,
        out_dir=out_dir,
        layer=args.layer,
        mask_var=args.mask_var,
        pseudotime_key=args.pseudotime_key,
        num_cells=args.num_cells,
        num_quantiles=args.num_quantiles,
        time_points=args.time_points,
        adj_list=adj_list,
        dt=args.dt,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        loss_func=args.loss_func,
        weighted_loss=args.weighted_loss,
        regularization=args.regularization,
        regularization_strength=args.regularization_strength,
        regularization_strength_min=args.regularization_strength_min,
        regularization_strength_max=args.regularization_strength_max,
        train_proportion=args.train_proportion,
        epochs_min=args.epochs_min,
        epochs_max=args.epochs_max,
        test_proportion=args.test_proportion,
        time_point_split=args.time_point_split,
        patience=args.patience,
    )

    # Run study
    run_optimization(
        objective_func=objective_func,
        out_dir=out_dir,
        study_name=args.name,
        dataset=args.input,
        num_cells=args.num_cells,
        num_quantiles=args.num_quantiles,
        loss_func=args.loss_func,
        weighted_loss=args.weighted_loss,
        regularization=args.regularization,
        regularization_strength=args.regularization_strength,
        train_proportion=args.train_proportion,
        test_proportion=args.test_proportion,
        time_point_split=args.time_point_split,
        n_trials=args.n_trials,
    )
