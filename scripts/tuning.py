import argparse
import logging
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal, Optional

import networkx as nx
import optuna
import pandas as pd
import scanpy as sc
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from bowside.dataset import PseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.training import mse_loss, qqloss, sinkhorn_loss, train_integration

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def objective(
    trial: optuna.Trial,
    input_path: str,
    out_dir: Path,
    layer: str,
    mask_var: str,
    pseudotime_key: str,
    adj_list: Optional[pd.DataFrame],
    dt: float,
    loss_func: str,
    weighted_loss: bool,
    regularization: Optional[Literal["l1", "l2"]],
    regularization_strength: Optional[float],
    seed: int,
):
    import diffrax
    import jax
    import optax
    from jax import numpy as jnp

    key = jax.random.fold_in(jax.random.key(seed), trial.number)
    # Setup per-trial logger
    checkpoint_dir = out_dir / str(trial.number)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    # Attach handler to sc_markov_ode.training logger
    logger = logging.getLogger(f"optuna_trial_{trial.number}")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info(f"Logger setup complete for trial {trial.number}")

    # Data loading
    adata = sc.read_h5ad(input_path)
    dataset = PseudotimeDataset(
        adata=adata, layer=layer, mask_var=mask_var, pseudotime_key=pseudotime_key
    )

    # Model parameters (encode booleans as strings for Optuna Dashboard)
    adj_init_normal = (
        trial.suggest_categorical("adj_init_normal", ["True", "False"]) == "True"
    )
    sigma = trial.suggest_float("sigma", 1e-3, 1e2, log=True)
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
        adj = jnp.asarray(nx.to_numpy_array(grn, nodelist=dataset.gene_names))

    # Optimizer setup
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optax, optimizer_name.lower())(learning_rate=lr)

    # Training parameters
    num_epochs = trial.suggest_int("num_epochs", low=500, high=5000, step=500)
    solver_method = trial.suggest_categorical("solver_method", ["Euler", "ItoMilstein"])

    key_init, key_train = jax.random.split(key)
    model = GRNModelSDE(
        dataset.n_genes(),
        adj=adj,
        adj_init_normal=adj_init_normal,
        sigma=sigma,
        shared_sigma=shared_sigma,
        fixed_sigma=fixed_sigma,
        constant_noise=constant_noise,
        solver=getattr(diffrax, solver_method)(),
        dt=dt,
        key=key_init,
    )

    # Regularization
    if regularization is not None:
        if regularization_strength is None:
            regularization_strength = trial.suggest_float(
                "regularization_strength", 1e-6, 1e-2, log=True
            )
    else:
        regularization_strength = (
            0.0001  # not used, but necessary for calling the function
        )

    # Training
    weight = None
    if weighted_loss:
        weight = jnp.sqrt(jnp.linspace(0.0, 10.0**2, dataset.time_points.size(0)))
    if loss_func == "sinkhorn":
        loss_fn = partial(sinkhorn_loss, weight=weight)
    elif loss_func == "qqloss":
        loss_fn = partial(
            qqloss,
            metric=mse_loss,
            dim=0,
            n_quantiles=100,
            weight=weight,
        )

    _, _, test_losses, _ = train_integration(
        dataset,
        model,
        optimizer,
        loss_fn,
        key_train,
        regularization=regularization,
        regularization_strength=regularization_strength,
        num_epochs=num_epochs,
        checkpoint="final",
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    logger.info(f"Final test loss: {test_losses[-1]}")
    file_handler.flush()
    logger.removeHandler(file_handler)
    file_handler.close()
    return test_losses[-1]


def run_optimization(
    _,
    objective_func,
    out_dir: Path,
    study_name: str,
    dataset: str,
    loss_func: str,
    weighted_loss: bool,
    regularization: Optional[Literal["l1", "l2"]],
    regularization_strength: Optional[float],
    n_trials: int,
):
    study = optuna.create_study(
        storage=JournalStorage(
            JournalFileBackend(file_path=str(out_dir / "journal.log"))
        ),
        study_name=study_name,
        load_if_exists=True,
    )
    study.set_user_attr("dataset", dataset)
    study.set_user_attr("loss_function", loss_func)
    study.set_user_attr("weighted_loss", str(weighted_loss))
    study.set_user_attr("regularization", regularization)
    study.set_user_attr("regularization_strength", regularization_strength)
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
        help="Adjacency matrix regularization strength. With the default `None`, treated as hyperparameter.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of trials for optuna tuning."
    )
    parser.add_argument(
        "--n-trials-per-worker",
        type=int,
        default=2,
        help="Number of trials to start on one worker.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel processes to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the initial, study-level JAX PRNG key.",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

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
        adj_list=adj_list,
        dt=args.dt,
        loss_func=args.loss_func,
        weighted_loss=args.weighted_loss,
        regularization=args.regularization,
        regularization_strength=args.regularization_strength,
        seed=args.seed,
    )

    # Run study
    with Pool(processes=args.n_jobs) as pool:
        pool.map(
            partial(
                run_optimization,
                objective_func=objective_func,
                out_dir=out_dir,
                study_name=args.name,
                dataset=args.input,
                loss_func=args.loss_func,
                weighted_loss=args.weighted_loss,
                regularization=args.regularization,
                regularization_strength=args.regularization_strength,
                n_trials=args.n_trials_per_worker,
            ),
            range(args.n_trials // args.n_trials_per_worker),
        )

    # Load finished study
    study = optuna.load_study(
        study_name=args.name,
        storage=JournalStorage(JournalFileBackend(str(out_dir / "journal.log"))),
    )

    # Print and save results
    print("Best parameters: ", study.best_params)
    print("Best value: ", study.best_value)
    print("Best trial: ", study.best_trial)
    print("Number of trials: ", len(study.trials))
    df: pd.DataFrame = study.trials_dataframe(
        attrs=("number", "value", "params", "state")
    )
    df.to_csv(out_dir / "trials_dataframe.csv")
