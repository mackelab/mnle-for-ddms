from typing import Optional

import numpy as np
import torch

from torch import Tensor, nn
from sbi.utils.sbiutils import standardizing_net
from torch.distributions import Bernoulli
from torch import Tensor



def build_choice_net(batch_theta, batch_choices, num_choices=2, z_score_theta=False, hidden_features: int=10, hidden_layers: int=2):
    
    dim_parameters = batch_theta[0].numel()
    num_output = num_choices
    
    assert num_choices == 2, "Not implemented for more than two choices."
    
    choice_net = BernoulliMN(n_input=dim_parameters, 
                             n_output=1,  # TODO: adapt to multiple choices.
                             n_hidden_layers=hidden_layers, 
                             n_hidden_units=hidden_features)
    
    if z_score_theta:
        choice_net = nn.Sequential(standardizing_net(batch_theta), choice_net)
    
    return choice_net

class MNLE(nn.Module):
    """Class for Mixed Neural Likelihood Estimation. It combines a Bernoulli choice
    net and a flow over reaction times to model decision-making data."""

    def __init__(
        self, choice_net: nn.Module, rt_net: nn.Module, use_log_rts: bool = False
    ):
        """Initializa synthetic likelihood class from a choice net and reaction time
        flow.

        Args:
            choice_net: BernoulliMN net trained to predict choices from DDM parameters.
            rt_net: generative model of reaction time given DDM parameters and choices.
            use_log_rts: whether the rt_net was trained with reaction times transformed
                to log space.
        """
        super(MNLE, self).__init__()

        self.choice_net = choice_net
        self.rt_net = rt_net
        self.use_log_rts = use_log_rts

    def sample(
        self,
        num_samples: int = 1,
        theta: Optional[Tensor] = None,
        track_gradients: bool = False,
    ) -> Tensor:
        """Return choices and reaction times given DDM parameters.

        Args:
            theta: DDM parameters, shape (batch, 4)
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples (rt, choice) with shape (num_samples, 2)
        """
        assert theta.shape[0] == 1, "for samples, no batching in theta is possible yet."

        with torch.set_grad_enabled(track_gradients):

            # Sample choices given parameters, from BernoulliMN.
            choices = self.choice_net.sample(num_samples, context=theta).reshape(
                num_samples, 1
            )
            # Pass num_samples=1 because the choices in the context contains num_samples elements already.
            rts = self.rt_net.sample(
                num_samples=1,
                # repeat the single theta to match number of sampled choices.
                context=torch.cat((theta.repeat(num_samples, 1), choices), dim=1),
            ).reshape(num_samples, 1)
            if self.use_log_rts:
                rts = rts.exp()

        return torch.cat((rts, choices), dim=1)

    def log_prob(
            self,
            x: Tensor,
            context: Tensor,
            track_gradients: bool = False,
            ll_lower_bound: float = -16.11,
        ) -> Tensor:
            """Return joint log likelihood of a batch rts and choices,for each entry in a
            batch of parameters theta.

            Note that x and theta are assumed to be repeated already, as required by the
            potential functions in the sbi package. Below torch.unique is used to get the
            original theta and x for efficient likelihood calculation.

            Note that we calculate the joint log likelihood over the batch of iid trials.
            Therefore, only theta can be batched and the data is fixed (or a batch of data
            is interpreted as iid trials)

            Args:
                x: the value to evaluate, typically a tensor containing reaction times and
                    choices, [rts; c].
                context: the context the values are conditioned on, typically parameters.
                track_gradients: whether to track gradients during evaluation, e.g., in HMC
                ll_lower_bound: lower bound on the returned log likelihoods.
            
            Returns:
                log_likelihood_trial_batch: log likelihoods for each trial and parameter.
            """
            assert x.shape[0] == context.shape[0], "x and context must have same batch size."
            # Extract unique values to undo trial-parameter-batch matching.
            theta = torch.unique(context, sorted=False, dim=0)
            num_parameters = theta.shape[0]
            x_unique = torch.unique(x, sorted=False, dim=0)
            num_trials = x_unique.shape[0]
            
            assert x_unique.ndim > 1
            assert (
                x_unique.shape[1] == 2
            ), "MNLE assumes x to have two columns: [rts; choices]"

            rts_repeated = x[:, 0:1]
            choices_repeated = x[:, 1:2]
            rts = x_unique[:, 0:1]
            choices = x_unique[:, 1:2]

            with torch.set_grad_enabled(track_gradients):
                # Get choice log probs from choice net.
                # There are only two choices, thus we only have to get the log probs of those.
                zero_choice = torch.zeros(1, 1)
                zero_choice_lp = self.choice_net.log_prob(
                    torch.repeat_interleave(zero_choice, num_parameters, dim=0),
                    context=theta,
                ).reshape(1, num_parameters)  # for each theta.

                # Calculate complement one-choice log prob.
                one_choice_lp = torch.log(1 - zero_choice_lp.exp())
                zero_one_lps = torch.cat((zero_choice_lp, one_choice_lp), dim=0)

                lp_choices = zero_one_lps[
                    choices.type_as(torch.zeros(1, dtype=np.int)).squeeze()
                ].reshape(-1)
                
                # Get rt log probs from rt net.
                lp_rts = self.rt_net.log_prob(
                    torch.log(rts_repeated) if self.use_log_rts else rts_repeated,
                    context=torch.cat((context, choices_repeated), dim=1),
                )

            # Combine into joint lp with first dim over trials.
            lp_combined = (lp_choices + lp_rts).reshape(num_trials, num_parameters)

            # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
            if self.use_log_rts:
                lp_combined -= torch.log(rts)
            # Set to lower bound where reaction happend before non-decision time tau.
            log_likelihood_trial_batch = torch.where(
                torch.logical_and(
                    # If rt < tau the likelihood should be zero (or at lower bound).
                    rts.repeat(1, num_parameters) > theta[:, -1],
                    # Apply lower bound.
                    lp_combined > ll_lower_bound,
                ),
                lp_combined,
                ll_lower_bound * torch.ones_like(lp_combined),
            )

            # Return batch over trials as required by SBI potentials.
            return log_likelihood_trial_batch


class BernoulliMN(nn.Module):
    """Net for learning a conditional Bernoulli mass function over choices given parameters.

    Takes as input parameters theta and learns the parameter p of a Bernoulli.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        n_input: int = 4,
        n_output: int = 1,
        n_hidden_units: int = 20,
        n_hidden_layers: int = 2,
    ):
        """Initialize Bernoulli mass network.

        Args:
            n_input: number of input features
            n_output: number of output features, default 1 for a single Bernoulli variable.
            n_hidden_units: number of hidden units per hidden layer.
            n_hidden_layers: number of hidden layers.
        """
        super(BernoulliMN, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        self.activation_fun = nn.Sigmoid()

        self.input_layer = nn.Linear(n_input, n_hidden_units)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        self.output_layer = nn.Linear(n_hidden_units, n_output)

    def forward(self, theta: Tensor) -> Tensor:
        """Return Bernoulli probability predicted from a batch of parameters.

        Args:
            theta: batch of input parameters for the net.

        Returns:
            Tensor: batch of predicted Bernoulli probabilities.
        """
        assert theta.dim() == 2, "theta needs to have a batch dimension."

        # forward path
        theta = self.activation_fun(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation_fun(layer(theta))

        return self.activation_fun(self.output_layer(theta))

    def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
        """Return Bernoulli log probability of choices x, given parameters theta.

        Args:
            theta: parameters for input to the BernoulliMN.
            x: choices to evaluate.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict Bernoulli p and evaluate.
        p = self.forward(context)

        return Bernoulli(probs=p).log_prob(x)

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """Returns samples from Bernoulli RV with p predicted via net.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Bernoulli samples with shape (batch, num_samples, 1)
        """

        # Predict Bernoulli p and sample.
        p = self.forward(context)
        return Bernoulli(probs=p).sample((num_samples,)).reshape(num_samples, -1)
