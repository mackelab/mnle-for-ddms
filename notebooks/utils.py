import math
import os
import pickle
from pathlib import Path
from typing import Any, Tuple

import lanfactory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sbi.inference.potentials.base_potential import BasePotential
from sbibm.utils.io import get_float_from_csv
from torch.distributions.transforms import AffineTransform
from tqdm.auto import tqdm


def compile_df(
    basepath: str,
) -> pd.DataFrame:
    """Compile dataframe for further analyses

    `basepath` is the path to a folder over which to recursively loop. All information
    is compiled into a big dataframe and returned for further analyses.

    Args:
        basepath: Base path to use

    Returns:
        Dataframe with results
    """
    df = []

    basepaths = [
        p.parent for p in Path(basepath).expanduser().rglob("posterior_samples.csv.bz2")
    ]

    for i, path_base in tqdm(enumerate(basepaths)):
        path_metrics = path_base / "metrics.csv"

        row = {}

        # Read hydra config
        path_cfg = path_metrics.parent / "run.yaml"
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        try:
            row["task"] = cfg["task"]["name"]
        except:
            continue
        row["num_simulations"] = cfg["task"]["num_simulations"]
        row["num_observation"] = cfg["task"]["num_observation"]
        row["algorithm"] = cfg["algorithm"]["name"]
        row["seed"] = cfg["seed"]

        # Metrics df
        if path_metrics.exists():
            metrics_df = pd.read_csv(path_metrics)
            for metric_name, metric_value in metrics_df.items():
                row[metric_name] = metric_value[0]
        else:
            continue

        # NLTP can be properly computed for NPE as part of the algorithm
        # SNPE's estimation of NLTP via rejection rates may introduce additional errors
        path_log_prob_true_parameters = (
            path_metrics.parent / "log_prob_true_parameters.csv"
        )
        row["NLTP"] = float("nan")
        if row["algorithm"][:3] != "NPE":
            if path_log_prob_true_parameters.exists():
                row["NLTP"] = torch.tensor(
                    get_float_from_csv(path_log_prob_true_parameters)
                )

        # Take log of KSD metric
        # NOTE: Since we originally did not log KSD, this is done post-hoc here
        row["KSD"] = math.log(row["KSD_GAUSS"])
        row["KSD_1K"] = math.log(row["KSD_GAUSS_1K"])
        del row["KSD_GAUSS"]
        del row["KSD_GAUSS_1K"]

        # Runtime
        # While almost all runs were executed on AWS hardware under the same conditions,
        # this was not the case for 100% of the runs. To prevent uneven comparison,
        # the file `runtime.csv` was deleted for those runs where this was not the case.
        # If `runtime.csv` is absent from a run, RT will be set to NaN accordingly.
        path_runtime = path_metrics.parent / "runtime.csv"
        if not path_runtime.exists():
            row["RT"] = float("nan")
        else:
            row["RT"] = torch.tensor(get_float_from_csv(path_runtime))

        # Runtime to minutes
        row["RT"] = row["RT"] / 60.0

        # Num simulations simulator
        path_num_simulations_simulator = (
            path_metrics.parent / "num_simulations_simulator.csv"
        )
        if path_num_simulations_simulator.exists():
            row["num_simulations_simulator"] = get_float_from_csv(
                path_num_simulations_simulator
            )

        # Path and folder
        row["path"] = str((path_base).absolute())
        row["folder"] = row["path"].split("/")[-1]

        # Exclude from df if there are no posterior samples
        if not os.path.isfile(f"{row['path']}/posterior_samples.csv.bz2"):
            continue

        df.append(row)

    df = pd.DataFrame(df)
    if len(df) > 0:
        df["num_observation"] = df["num_observation"].astype("category")

    return df


# affine transform
# add scale transform for "a": to go to LAN prior space, i.e., multiply with scale 0.5.
a_transform = AffineTransform(torch.zeros(1, 4), torch.tensor([[1.0, 0.5, 1.0, 1.0]]))

# LAN potential function
class LANPotential(BasePotential):
    allow_iid_x = True  # type: ignore
    """Inits LAN potential as required by the sbi package for MCMC sampling."""

    def __init__(
        self,
        lan,
        prior: torch.distributions.Distribution,
        x_o: torch.Tensor,
        device: str = "cpu",
        apply_a_transform: bool = False,
        ll_lower_bound: float = np.log(1e-7),
        apply_ll_lower_bound: bool = False,
    ):
        """Init.

        Args:
            lan: pretrained LAN holding the pytorch module.
            prior: prior for inference
            x_o: observation, can be multi-trial.
            device: device of the LAN
            apply_a_transform: whether to transform the DDM parameter 'a' to go
                from symmetric boundary width (LAN), to boundary separation (SBI)
            ll_lower_bound: lower bound on the log-likelihood evaluations,
                as used in the LAN paper.
            apply_ll_lower_bound: whether to enforce the lower bound.
        """
        super().__init__(prior, x_o, device)

        self.lan = lan
        self.device = device
        self.ll_lower_bound = ll_lower_bound
        self.apply_ll_lower_bound = apply_ll_lower_bound
        self.apply_a_transform = apply_a_transform

        # make sure data is encoded in 1D: choices as sign of rt.
        assert x_o.ndim == 2
        assert x_o.shape[1] == 1
        # decode to separate rt and choices (-1, 1).
        rts = abs(x_o)
        num_trials = rts.numel()
        assert rts.shape == torch.Size([num_trials, 1])
        # Code down -1 up +1.
        cs = torch.ones_like(rts)
        cs[x_o < 0] *= -1

        self.num_trials = num_trials
        self.rts = rts
        self.cs = cs

    def __call__(self, theta, track_gradients=False):
        """Return potential function value for a batch of parameters theta."""

        num_parameters = theta.shape[0]
        # Convert DDM boundary seperation to symmetric boundary size.
        # theta_lan = a_transform(theta)
        theta_lan = a_transform(theta) if self.apply_a_transform else theta

        # Evaluate LAN on batch (as suggested in LANfactory README.)
        batch = torch.hstack(
            (
                theta_lan.repeat(self.num_trials, 1),  # repeat params for each trial
                self.rts.repeat_interleave(
                    num_parameters, dim=0
                ),  # repeat data for each param
                self.cs.repeat_interleave(num_parameters, dim=0),
            )
        )
        log_likelihood_trials = self.lan(batch).reshape(self.num_trials, num_parameters)

        # Sum over trials.
        # Lower bound on each trial log likelihood.
        # Sum across trials.
        if self.apply_ll_lower_bound:
            log_likelihood_trial_sum = (
                torch.where(
                    torch.logical_and(
                        self.rts.repeat(1, num_parameters) > theta[:, -1],
                        log_likelihood_trials > self.ll_lower_bound,
                    ),
                    log_likelihood_trials,
                    self.ll_lower_bound * torch.ones_like(log_likelihood_trials),
                )
                .sum(0)
                .squeeze()
            )
        else:
            log_likelihood_trial_sum = log_likelihood_trials.sum(0).squeeze()

        # Maybe apply correction for transform on "a" parameter.
        if self.apply_a_transform:
            log_abs_det = a_transform.log_abs_det_jacobian(theta_lan, theta)
            if log_abs_det.ndim > 1:
                log_abs_det = log_abs_det.sum(-1)
            log_likelihood_trial_sum -= log_abs_det

        return log_likelihood_trial_sum + self.prior.log_prob(theta)


def lan_likelihood_on_batch(
    data: torch.Tensor,
    theta: torch.Tensor,
    net,
    transform,
    device,
):
    """Return LAN log-likelihood given a batch of data and parameters.

    Return shape: , (batch_size_data, batch_size_parameters)

    """
    # Convert to positive rts.
    rts = abs(data)
    num_trials = rts.numel()
    num_parameters = theta.shape[0]
    assert rts.shape == torch.Size([num_trials, 1])
    theta = torch.tensor(theta, dtype=torch.float32)
    # Convert DDM boundary seperation to symmetric boundary size.
    theta_lan = transform(theta)

    # Code down -1 up +1.
    cs = torch.ones_like(rts)
    cs[data < 0] *= -1

    # Evaluate LAN on batch (as suggested in LANfactory README.)
    batch = torch.hstack(
        (
            theta_lan.repeat(num_trials, 1),  # repeat params for each trial
            rts.repeat_interleave(num_parameters, dim=0),  # repeat data for each param
            cs.repeat_interleave(num_parameters, dim=0),
        )
    )
    log_likelihood_trials = net(batch.to(device)).reshape(num_trials, num_parameters)

    return log_likelihood_trials.to("cpu")


def apply_lower_bound_given_mask(ll, mask, ll_lower_bound: float = np.log(1e-7)):
    """Replaces values at mask with lower bound."""

    assert mask.shape == ll.shape, "Mask must have the same shape as the input."

    ll[mask] = ll_lower_bound

    return ll


def decode_1d_to_2d_x(x1d):
    """Decodes rts with choices encoded as sign into (rts, 0-1-choices)"""
    x = torch.zeros((x1d.shape[0], 2))
    # abs rts in first column
    x[:, 0] = abs(x1d[:, 0])
    # 0 - 1 code for choices in second column.
    x[x1d[:, 0] > 0, 1] = 1

    return x


# Define losses.
def huber_loss(y, yhat):
    diff = abs(y - yhat)

    err = np.zeros(y.numel())
    err[diff <= 1.0] = 0.5 * diff[diff <= 1.0] ** 2
    err[diff > 1.0] = 0.5 + diff[diff > 1.0]
    return err.mean()


def mean_squared_error(y, yhat):
    return torch.mean((y - yhat) ** 2)


def load_lan(budget: str) -> Tuple(str, Any):
    """Returns a pretrained LAN given a certain simulation budget."""

    lan_model_folder = (
        Path.cwd() / f"mnle-lan-comparison/data/torch_models/ddm_{budget}/"
    )  # Pathlib object
    network_file_path = list(
        lan_model_folder.glob(f"*_ddm_{budget}_torch_state_dict.pt")
    )[0]

    # get network config from model folder.
    with open(
        list(network_file_path.parent.glob("*_network_config.pickle"))[0], "rb"
    ) as fh:
        network_config = pickle.load(fh)

    # load model
    network = lanfactory.trainers.LoadTorchMLPInfer(
        model_file_path=network_file_path, network_config=network_config, input_dim=6
    )  # 4 params plus 2 data dims

    return network_file_path, network


def plot_bar(pos, width, metrics, label, color, alpha=0.6):
    """Helper function for bar plots below."""
    plt.bar(
        pos,
        height=metrics.mean(0),
        width=width,
        yerr=metrics.std(0) / np.sqrt(metrics.shape[0]),
        color=color,
        label=label,
        alpha=alpha,
    )
