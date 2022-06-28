import lanfactory
import numpy as np
import sbibm
import torch

from torch.distributions.transforms import AffineTransform

from sbi.inference.potentials.base_potential import BasePotential

# affine transform
# add scale transform for "a": to go to LAN prior space, i.e., multiply with scale 0.5.
a_transform = AffineTransform(torch.zeros(1, 4), torch.tensor([[1.0, 0.5, 1.0, 1.0]]))

# LAN potential function
class LANPotential(BasePotential):
    allow_iid_x = True  # type: ignore
    """Inits LAN potential."""

    def __init__(self, 
            lan, 
            prior: torch.distributions.Distribution, 
            x_o: torch.Tensor, 
            device: str="cpu", 
            apply_a_transform: bool=False, 
            ll_lower_bound: float=np.log(1e-7), 
            apply_ll_lower_bound: bool = False
        ):
        super().__init__(prior, x_o, device)

        self.lan = lan
        self.device = device
        self.ll_lower_bound = ll_lower_bound
        self.apply_ll_lower_bound = apply_ll_lower_bound
        self.apply_a_transform = apply_a_transform
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
        theta_lan = a_transform(theta) if self.apply_a_transform else theta

        # Evaluate LAN on batch (as suggested in LANfactory README.)
        batch = torch.hstack((
            theta_lan.repeat(self.num_trials, 1),  # repeat params for each trial
            self.rts.repeat_interleave(num_parameters, dim=0),  # repeat data for each param
            self.cs.repeat_interleave(num_parameters, dim=0))
        )
        log_likelihood_trials = self.lan(batch).reshape(self.num_trials, num_parameters)

        # Sum over trials.
        # Lower bound on each trial log likelihood.
        # Sum across trials.
        if self.apply_ll_lower_bound:
            log_likelihood_trial_sum = torch.where(
                torch.logical_and(
                    self.rts.repeat(1, num_parameters) > theta[:, -1], 
                    log_likelihood_trials > self.ll_lower_bound,
                ),
                log_likelihood_trials,
                self.ll_lower_bound * torch.ones_like(log_likelihood_trials),
            ).sum(0).squeeze()
        else:
            log_likelihood_trial_sum = log_likelihood_trials.sum(0).squeeze()

        # Maybe apply correction for transform on "a" parameter.
        if self.apply_a_transform:
            log_abs_det = a_transform.log_abs_det_jacobian(theta_lan, theta)
            if log_abs_det.ndim > 1:
                    log_abs_det = log_abs_det.sum(-1)
            log_likelihood_trial_sum -= log_abs_det

        return log_likelihood_trial_sum + self.prior.log_prob(theta)

class OldLANPotential(BasePotential):
    allow_iid_x = True  # type: ignore
    """Inits LAN potential."""

    def __init__(self, 
            lan, 
            prior: torch.distributions.Distribution, 
            x_o: torch.Tensor, 
            device: str="cpu", 
            apply_a_transform: bool=False, 
            ll_lower_bound: float=np.log(1e-7), 
            apply_ll_lower_bound: bool = False):
        super().__init__(prior, x_o, device)

        self.lan = lan
        self.device = device
        self.ll_lower_bound = ll_lower_bound
        self.apply_ll_lower_bound = apply_ll_lower_bound
        self.apply_a_transform = apply_a_transform
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
        theta_lan = a_transform(theta) if self.apply_a_transform else theta

        # Evaluate LAN on batch (as suggested in LANfactory README.)
        batch = torch.hstack((
            theta_lan.repeat(self.num_trials, 1),  # repeat params for each trial
            self.rts.repeat_interleave(num_parameters, dim=0),  # repeat data for each param
            self.cs.repeat_interleave(num_parameters, dim=0))
        )
        log_likelihood_trials = torch.tensor(
            self.lan.predict_on_batch(batch.numpy()),
            dtype=torch.float32,
        ).reshape(self.num_trials, num_parameters)
        

        # Sum over trials.
        # Lower bound on each trial log likelihood.
        # Sum across trials.
        if self.apply_ll_lower_bound:
            log_likelihood_trial_sum = torch.where(
                torch.logical_and(
                    self.rts.repeat(1, num_parameters) > theta[:, -1], 
                    log_likelihood_trials > self.ll_lower_bound,
                ),
                log_likelihood_trials,
                self.ll_lower_bound * torch.ones_like(log_likelihood_trials),
            ).sum(0).squeeze()
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
    data, 
    theta, 
    net, 
    transform,
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

    batch = torch.hstack((theta_lan, rts.repeat(num_parameters, 1), cs.repeat(num_parameters, 1)))

    # Evaluate LAN on batch (as suggested in LANfactory README.)    
    ll_each_trial = net(batch).reshape(num_trials, num_parameters)
    
    return ll_each_trial
    

def apply_lower_bound_given_mask(ll, mask, ll_lower_bound: float=np.log(1e-7)):
    """Replaces values at mask with lower bound."""

    assert mask.shape == ll.shape, "Mask must have the same shape as the input."

    ll[mask] = ll_lower_bound

    return ll

def decode_1d_to_2d_x(x1d):
    """Decodes rts with choices encoded as sign into (rts, 0-1-choices) """
    x = torch.zeros((x1d.shape[0], 2))
    # abs rts in first column
    x[:, 0] = abs(x1d[:, 0])
    # 0 - 1 code for choices in second column.
    x[x1d[:, 0] > 0, 1] = 1
    
    return x

# Define losses.
def huber_loss(y, yhat):
    diff = abs(y-yhat)
    
    err = np.zeros(y.numel())
    err[diff <= 1.0] = 0.5 * diff[diff <= 1.0]**2
    err[diff > 1.0] = 0.5 + diff[diff > 1.0]
    return err.mean()

def mean_squared_error(y, yhat):
    return torch.mean((y - yhat)**2)