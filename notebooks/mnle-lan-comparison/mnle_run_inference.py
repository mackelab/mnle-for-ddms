import pickle
from pathlib import Path

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


# load a LAN
budget = "100000"
model_path = Path.cwd() / f"models/"
network_file_path = list(model_path.glob(f"mnle_n{budget}_new*"))[0]  # take first model from random inits.

with open(network_file_path, "rb") as fh:
    mnle, *_ = pickle.load(fh).values()

mcmc_parameters = dict(
    warmup_steps = 100, 
    thin = 10, 
    num_chains = 10,
    num_workers = 10,
    init_strategy = "sir",
    )

# Build MCMC posterior in SBI.
mnle_posterior = MNLE().build_posterior(mnle, prior, 
    mcmc_method="slice_np_vectorized", 
    mcmc_parameters=mcmc_parameters,
    )

samples = []
num_samples = 1000
for x_o in xos_2d:

    mnle_samples = mnle_posterior.sample(
        (num_samples,), 
        x=x_o.reshape(100, 2)
    )
    
    samples.append(mnle_samples)

with open(f"mnle_{budget}_posterior_samples_{num_obs}x100iid.p", "wb") as fh:
    pickle.dump(samples, fh)

