# Script to run inference with MCMC given a pretrained LAN
# Uses MCMC methods from the sbi package, via a custom potential 
# function wrapper in utils.

import pickle
from pathlib import Path
from joblib import Parallel, delayed

import lanfactory
import sbibm
import torch

from sbi.inference import MCMCPosterior
from sbi.utils import mcmc_transform

from utils import LANPotential


BASE_DIR = Path.cwd().parent.parent
save_folder = BASE_DIR / "data/results"

# Get benchmark task to load observations
seed = torch.randint(100000, (1,)).item()

task = sbibm.get_task("ddm")
prior = task.get_prior_dist()
simulator = task.get_simulator(seed=seed) # Passing the seed to Julia.

# Observation indices >200 hold 100-trial observations
num_trials = 100
if num_trials == 1:
    start_obs = 0
elif num_trials == 10:
    start_obs = 100
elif num_trials == 100:
    start_obs = 200
else:
    start_obs = 300

num_obs = 100
xos = torch.stack([task.get_observation(start_obs + ii) for ii in range(1, 1+num_obs)]).reshape(num_obs, num_trials)

# encode xos as (time, choice)
xos_2d = torch.zeros((xos.shape[0], xos.shape[1], 2))
for idx, xo in enumerate(xos):
    xos_2d[idx, :, 0] = abs(xo)
    xos_2d[idx, xo > 0, 1] = 1


# load a LAN
budget = "10_11"
apply_a_transform = True
apply_ll_lower_bound = True
model_path = Path.cwd() / f"data/torch_models/ddm_{budget}/"
network_file_path = list(model_path.glob("*state_dict*"))[0]  # take first model from random inits.

# get network config from model folder.
with open(list(network_file_path.parent.glob("*_network_config.pickle"))[0], "rb") as fh:
    network_config = pickle.load(fh)

# load model
lan = lanfactory.trainers.LoadTorchMLPInfer(model_file_path = network_file_path,
                                            network_config = network_config,
                                            input_dim = 6)  # 4 params plus 2 data dims)

# load old LAN
from tensorflow import keras
# network trained on KDE likelihood for 4-param ddm
lan_kde_path = Path.cwd() / "../../data/pretrained-models/model_final_ddm.h5"
lan = keras.models.load_model(lan_kde_path, compile=False)



mcmc_parameters = dict(
    warmup_steps = 100, 
    thin = 10, 
    num_chains = 10,
    num_workers = 1,
    init_strategy = "sir",
    )

# Build MCMC posterior in SBI.
theta_transform = mcmc_transform(prior)

samples = []
num_samples = 10000
num_workers = 20

def run(x_o):
    lan_potential = LANPotential(lan, 
        prior, 
        x_o.reshape(-1, 1), 
        apply_a_transform=apply_a_transform, 
        apply_ll_lower_bound=apply_ll_lower_bound,
        )
    lan_posterior = MCMCPosterior(lan_potential,
        proposal=prior,
        theta_transform=theta_transform, 
        method="slice_np_vectorized",
        **mcmc_parameters,
        )
    
    return lan_posterior.sample((num_samples,), x=x_o)

results = Parallel(n_jobs=num_workers)(
    delayed(run)(x_o) for x_o in xos
)

with open(save_folder / f"lan_{budget}_posterior_samples_{num_obs}x{num_trials}iid_old.p", "wb") as fh:
    pickle.dump(results, fh)
