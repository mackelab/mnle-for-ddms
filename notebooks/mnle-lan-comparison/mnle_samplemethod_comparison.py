import pickle
from pathlib import Path
import time

import sbibm
import torch

from sbi.inference import MNLE

# Get benchmark task to load observations
seed = torch.randint(100000, (1,)).item()

task = sbibm.get_task("ddm")
prior = task.get_prior_dist()
simulator = task.get_simulator(seed=seed) # Passing the seed to Julia.

# Observation indices >200 hold 100-trial observations
num_obs = 100
xos = torch.stack([task.get_observation(200 + ii) for ii in range(1, 1+num_obs)]).squeeze()

# encode xos as (time, choice)
xos_2d = torch.zeros((xos.shape[0], xos.shape[1], 2))
for idx, xo in enumerate(xos):
    xos_2d[idx, :, 0] = abs(xo)
    xos_2d[idx, xo > 0, 1] = 1

BASE_DIR = Path.cwd().parent.parent
save_folder = BASE_DIR / "data/results"

# load a LAN
budget = "100000"
model_path = Path.cwd() / f"models/"
network_file_path = list(model_path.glob(f"mnle_n{budget}_new*"))[5]  # take first model from random inits.

with open(network_file_path, "rb") as fh:
    mnle, *_ = pickle.load(fh).values()

mcmc_parameters = dict(
    warmup_steps = 100, 
    thin = 10, 
    num_chains = 10,
    num_workers = 1,
    init_strategy = "sir",
    )

# Build MCMC posterior in SBI.

num_samples = 10000
obs_idx = 10
xo = xos_2d[obs_idx]

reference_samples = task.get_reference_posterior_samples(201+obs_idx)[:num_samples]
true_theta = task.get_true_parameters(201+obs_idx)

## NUTS
tic = time.time()
mnle_posterior = MNLE().build_posterior(mnle, prior, 
    mcmc_method="nuts", 
    mcmc_parameters=mcmc_parameters,
    )

nuts_samples = mnle_posterior.sample((num_samples,), x=xo, mp_context="fork")
nuts_time = time.time() - tic

## Slice sampling
tic = time.time()
mnle_posterior = MNLE().build_posterior(mnle, prior, 
    mcmc_method="slice_np_vectorized", 
    mcmc_parameters=mcmc_parameters,
    )

slice_samples = mnle_posterior.sample((num_samples,), x=xo)
slice_time = time.time() - tic

## VI
tic = time.time()
viposterior = MNLE().build_posterior(mnle, prior, 
    sample_with="vi"
    )
viposterior.set_default_x(xo)
viposterior.train()
vi_samples = viposterior.sample((num_samples, ))
vi_time = time.time() - tic

with open(save_folder / f"mnle_samplemethod_comparison_obs{201+obs_idx}.p", "wb") as fh:
    pickle.dump(dict(
        reference_samples=reference_samples, 
        true_theta=true_theta,
        slice_samples=slice_samples,
        vi_samples=vi_samples,
        nuts_samples=nuts_samples,
        timings=dict(vi=vi_time, slice=slice_time, nuts=nuts_time),
        ), fh)
