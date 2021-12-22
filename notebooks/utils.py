import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deneb.utils import rgb2hex
from omegaconf import OmegaConf
from sbibm.utils.io import get_float_from_csv
from tqdm.auto import tqdm

def compile_df(basepath: str,) -> pd.DataFrame:
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

# Define losses.
def huber_loss(y, yhat):
    diff = abs(y-yhat)
    
    err = np.zeros(y.numel())
    err[diff <= 1.0] = 0.5 * diff[diff <= 1.0]**2
    err[diff > 1.0] = 0.5 + diff[diff > 1.0]
    return err.mean()

def mean_squared_error(y, yhat):
    return torch.mean((y - yhat)**2)