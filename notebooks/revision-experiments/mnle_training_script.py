# Script for training MNLE with pre-simulated data.

import pickle
import torch

from joblib import Parallel, delayed
from pathlib import Path
from sbi.inference import MNLE
from sbi.utils import likelihood_nn


BASE_DIR = Path(__file__).resolve().parent.parent.parent.as_posix()
data_folder = BASE_DIR + "/data/"
save_folder = BASE_DIR + "/notebooks/mnle-lan-comparison/models/"

# MNLE settings
mnle_provider = likelihood_nn(
    model="mnle",
    **dict(
        log_transform_x=False,
        num_bins=5,
        num_transforms=2,
        tail_bound=10.0,
        hidden_layers=1,
        hidden_features=10,
    ),
)
batch_size = 100
stop_after_epochs = 30


def train_mnle(theta, x, num_simulations, seed):
    """Returns trained MNLE of type MixedNeuralLikelihoodEstimator (nn.Module)."""

    theta = theta[:num_simulations]
    x = x[:num_simulations]

    trainer = MNLE(density_estimator=mnle_provider)
    estimator = trainer.append_simulations(theta, x).train(
        training_batch_size=batch_size, stop_after_epochs=stop_after_epochs
    )

    with open(save_folder + f"mnle_n{num_simulations}_new_seed{seed}.p", "wb") as fh:
        pickle.dump(dict(estimator=estimator, num_simulations=num_simulations), fh)

    return estimator


# Load pre-simulated training data
with open(data_folder + "ddm_training_and_test_data_10mio.p", "rb") as fh:
    theta, x_1d, test_x, test_thetas = pickle.load(fh).values()

# encode x as (time, choice)
x = torch.zeros((x_1d.shape[0], 2))
x[:, 0] = abs(x_1d[:, 0])
x[x_1d[:, 0] > 0, 1] = 1

num_workers = 10
num_repeats = 10
budgets = torch.tensor([200_000]).repeat_interleave(num_repeats)
seeds = torch.randint(0, 1000000, size=(budgets.shape[0],))

results = Parallel(n_jobs=num_workers)(
    delayed(train_mnle)(theta, x, budget.item(), seed)
    for budget, seed in zip(budgets, seeds)
)
