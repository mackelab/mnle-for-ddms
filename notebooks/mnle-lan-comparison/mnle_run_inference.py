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
num_trials = 1
if num_trials == 1:
    start_obs = 0
elif num_trials == 10:
    start_obs = 100
elif num_trials == 100:
    start_obs = 200
else:
    start_obs = 300

num_obs = 100
xos = torch.stack([task.get_observation(200 + ii) for ii in range(1, 1+num_obs)]).squeeze()

# encode xos as (time, choice)
xos_2d = torch.zeros((xos.shape[0], xos.shape[1], 2))
for idx, xo in enumerate(xos):
    xos_2d[idx, :, 0] = abs(xo)
    xos_2d[idx, xo > 0, 1] = 1


# load a LAN
budget = "1000000"
model_path = Path.cwd() / f"models/"
init_idx = 1
network_file_path = list(model_path.glob(f"mnle_n{budget}_new*"))[init_idx]  # take one model from random inits.

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
    # sample_with="vi"
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

    # mnle_posterior.set_default_x(x_o.reshape(100, 2))
    # mnle_posterior.train()
    # vi_samples = mnle_posterior.sample((num_samples, ))
    
    samples.append(mnle_samples)

with open(f"mnle-{init_idx}_{budget}_posterior_samples_{num_obs}x{num_trials}*iid.p", "wb") as fh:
    pickle.dump(samples, fh)

