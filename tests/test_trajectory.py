import anndata as ad
import pandas as pd
import pytest
import torch

from bowside.dataset import MarkovChainDataset
from bowside.model import GRNModelSDE
from bowside.training import integrate_model
from bowside.trajectory import infer_trajectories


class BifurcatingSDE(GRNModelSDE):
    """
    Dummy SDE generating a bifurcating system with two genes.
    """

    def __init__(self):
        super().__init__(2)
        adj = torch.tensor([[2, -2.0], [-2.0, 2.0]])
        bias = torch.tensor([1.0, 1.0])
        decay_rate = torch.tensor([0.6, 0.6])
        with torch.no_grad():
            self.adj.copy_(adj.float())
            self.log_bias.copy_(torch.log(bias.float()))
            self.log_decay_rate.copy_(torch.log(decay_rate.float()))


class BifurcatingDataset(MarkovChainDataset):
    """
    Dummy dataset simulating an SDE with a defined adjacency matrix
    instead of using a single-cell dataset.
    """

    def __init__(self, y0: torch.Tensor):
        self.sde = BifurcatingSDE()
        n_steps = 50
        self.time_points = torch.linspace(
            0, n_steps - 1, n_steps, dtype=torch.float32, device=y0.device
        )
        self.simulations = integrate_model(
            y0, self.sde, time_points=self.time_points, dt=0.1
        ).permute((1, 0, 2))
        self.adata = ad.AnnData()
        self.gene_names = pd.Index(
            [f"gene_{i}" for i in range(self.simulations.size(2))]
        )

    def __len__(self):
        return self.simulations.shape[0]

    def __getitem__(self, idx):
        return self.simulations[idx, :, :]


@pytest.mark.parametrize(
    "y0",
    [
        torch.tensor([[0.0, 0.0]]).repeat(500, 1),
        torch.tensor([[0.0, 0.3]]).repeat(500, 1),
    ],
)
def test_trajectory_inference(y0: torch.Tensor):
    dataset = BifurcatingDataset(y0)
    model = dataset.sde
    infer_trajectories(dataset, model, dt=0.1)

    trajectory_inference_results = dataset.adata.uns["bowside"]
    num_trajectories = trajectory_inference_results["num_trajectories"]
    simulations = trajectory_inference_results["simulations"]

    assert num_trajectories == 2, (
        f"There should be 2 trajectories, but there are {num_trajectories}."
    )

    for i in range(dataset.n_genes()):
        gene_i_up_trajectory = trajectory_inference_results["trajectories"][
            :, -1, i
        ].argmax()
        final_expression = simulations[-1, :, i]
        cluster_indices = (
            trajectory_inference_results["trajectory_inference_clusters"]
            == gene_i_up_trajectory + 1
        )
        assert (
            final_expression[cluster_indices].min()
            >= final_expression[~cluster_indices].max()
        ), (
            f"The final expression of gene {i} should be higher in all simulations "
            "of its upregulated trajectory than in any other simulation."
        )
