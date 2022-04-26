import pickle
from pathlib import Path
from joblib import Parallel, delayed

import lanfactory
import numpy as np
import sbibm
import torch

from torch.distributions.transforms import AffineTransform

from sbi.inference import MCMCPosterior
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import mcmc_transform


BASE_DIR = Path.cwd().parent.parent
save_folder = BASE_DIR / "data/results"

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
budget = "10_8_ours"
model_path = Path.cwd() / f"data/torch_models/ddm_{budget}/"
network_file_path = list(model_path.glob("*state_dict*"))[0]  # take first model from random inits.

# get network config from model folder.
with open(list(network_file_path.parent.glob("*_network_config.pickle"))[0], "rb") as fh:
    network_config = pickle.load(fh)

# load model
lan = lanfactory.trainers.LoadTorchMLPInfer(model_file_path = network_file_path,
                                            network_config = network_config,
                                            input_dim = 6)  # 4 params plus 2 data dims)

# LAN potential function
class LANPotential(BasePotential):
    allow_iid_x = True  # type: ignore
    """Inits LAN potential."""

    def __init__(self, lan, prior, x_o, device="cpu", ll_lower_bound=np.log(1e-7)):
        super().__init__(prior, x_o, device)

        self.lan = lan
        self.device = device
        self.ll_lower_bound = ll_lower_bound
        assert x_o.ndim == 2
        assert x_o.shape[1] == 1    
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
            
        num_parameters = theta.shape[0]
        # Convert DDM boundary seperation to symmetric boundary size.
        # theta_lan = a_transform(theta)
        theta_lan = theta

        # Evaluate LAN on batch (as suggested in LANfactory README.)
        batch = torch.hstack((
            theta_lan.repeat(self.num_trials, 1),  # repeat params for each trial
            self.rts.repeat_interleave(num_parameters, dim=0),  # repeat data for each param
            self.cs.repeat_interleave(num_parameters, dim=0))
        )
        log_likelihood_trials = lan(batch).reshape(self.num_trials, num_parameters)

        # Sum over trials.
        log_likelihood_trial_sum = log_likelihood_trials.sum(0).squeeze()

        # Apply correction for transform on "a" parameter.
        # log_abs_det = a_transform.log_abs_det_jacobian(theta_lan, theta)
        # if log_abs_det.ndim > 1:
        #         log_abs_det = log_abs_det.sum(-1)
        # log_likelihood_trial_sum -= log_abs_det

        return log_likelihood_trial_sum + self.prior.log_prob(theta)

mcmc_parameters = dict(
    warmup_steps = 100, 
    thin = 10, 
    num_chains = 10,
    num_workers = 1,
    init_strategy = "sir",
    )

# Build MCMC posterior in SBI.
theta_transform = mcmc_transform(prior)
# affine transform
# add scale transform for "a": to go to LAN prior space, i.e., multiply with scale 0.5.
a_transform = AffineTransform(torch.zeros(1, 4), torch.tensor([[1.0, 0.5, 1.0, 1.0]]))

samples = []
num_samples = 1000
num_workers = 10

def run(x_o):
    lan_posterior = MCMCPosterior(LANPotential(lan, prior, x_o.reshape(-1, 1)),
        proposal=prior,
        theta_transform=theta_transform, 
        method="slice_np_vectorized",
        **mcmc_parameters,
        )
    
    return lan_posterior.sample((num_samples,), x=x_o)

results = Parallel(n_jobs=num_workers)(
    delayed(run)(x_o) for x_o in xos
)

with open(save_folder / f"lan_{budget}_posterior_samples_{num_obs}x100iid.p", "wb") as fh:
    pickle.dump(results, fh)
