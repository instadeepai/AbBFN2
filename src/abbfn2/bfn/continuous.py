# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import distrax
import jax.numpy as jnp
from distrax import Normal
from jax import Array
from jax.random import PRNGKey

from abbfn2.bfn.base import BFNBase
from abbfn2.bfn.types import OutputNetworkPredictionContinuous, ThetaContinuous


class ContinuousBFN(BFNBase):
    """Continuous-variable Bayesian Flow Network."""

    def get_prior_input_distribution(self) -> ThetaContinuous:
        """Initialises the parameters of an uninformed input distribution."""
        mu = jnp.zeros(self.cfg.variables_shape)
        rho = jnp.ones(self.cfg.variables_shape)
        return ThetaContinuous(mu=mu, rho=rho)

    def sample_sender_distribution(
        self,
        x: Array,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a noise sample for the ground-truth (x) from the sender distribution.

        Specifically, this function samples y ~ p_S(y|x;α) = N(y|x,I/α).

        Args:
            x (Array): The ground truth data of shape [...var_shape...].
            alpha (Array): A per-variable accuracy parameter (shape [...var_shape...]).
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the sender distribution (shape [...var_shape...]).

        Notes:
            The equations implemented here are described in eq 86 of Graves et al.
        """
        # y ~ p_S(y|x;α) = N(y|x,I/α) --> y = x + z/α; z ~ N(0,I)
        z = distrax.Normal(0, 1).sample(seed=key, sample_shape=x.shape)
        y = x + (z / jnp.sqrt(alpha))  # Note: Normal has scale = s.d. hence sqrt.
        return y

    def sample_receiver_distribution(
        self,
        pred: OutputNetworkPredictionContinuous,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a sample from the receiver distribution.

        Specifically, this function samples y ~ p_R(y|θ;t,α) = N(y|x(θ,t),I/α).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            alpha (Array): A per-variable accuracy parameter (shape [...var_shape...]).
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the receiver distribution (shape [...var_shape...]).

        Notes:
            - The equations implemented here are described in eq 87-88 of Graves et al.
            - As it has the same form as the sampling of the sender distribution, we re-use
              this function internally.
        """
        return self.sample_sender_distribution(pred.x, alpha, key)

    def update_distribution(
        self,
        theta: ThetaContinuous,
        y: Array,
        alpha: Array | None = None,
        conditional_score: Array | None = None,
        conditional_mask: Array | None = None,
    ) -> ThetaContinuous:
        """Apply update to distribution parameters given sample of receiver distribution.

        Args:
            theta (ThetaContinuous): Parameters of the distribution.
            y (Array): The sample from the sender distribution (shape [...var_shape...]).
            alpha (Array): A per-variable accuracy parameter (shape [...var_shape...]).
            conditional_score (Optional[Array]): Per-variable conditional score
                (gradient of log prob of conditional data wrt input parameters)
                used to update the input parameters for conditional SDE sampling.
                If None, update reverts to unconditional.
            conditional_mask (Optional[Array]): Per-variable mask for conditional sampling.
                The conditional update only happens where the mask is False
                (no need if the mask is True since the ground truth is already known).

        Returns:
            theta (ThetaContinuous): Updated parameters of the distribution.
        """
        mu = (theta.rho * theta.mu + alpha * y) / (theta.rho + alpha)
        rho = theta.rho + alpha
        if conditional_score is not None:
            old_variance = 1 / theta.rho
            new_variance = 1 / rho
            mu_delta = (old_variance - new_variance) * conditional_score.mu
            if conditional_mask is not None:
                mu_delta = jnp.where(conditional_mask, 0, mu_delta)
            mu += mu_delta

        return ThetaContinuous(mu=mu, rho=rho)

    def conditional_log_prob(
        self,
        pred: OutputNetworkPredictionContinuous,
        x: Array | None,
        mask: Array | None,
        theta: ThetaContinuous,
    ) -> float:
        """Calculate log p(x|theta) (used to determine the conditional score for SDE sampling).

        Args:
            pred (OutputNetworkPredictionContinuous): Prediction of the output network.
            x(Optional[Array]): Conditioning data.
            mask (Optional[Array]): Per-variable boolean mask for the conditioning data.
                    Valid masks can be broadcast the shape of x and are True (False) if a conditional variable is used (unused).
            theta (ThetaContinuous): Parameters of the input distribution.


        Returns:
            The summed log prob over all the variables in x where mask=True. Returns 0 if x is None
        """
        if x is None:
            return 0
        else:
            mean = pred.x
            std_dev = 1.0 / jnp.sqrt(theta.rho)
            output_distribution = Normal(mean, std_dev)
            log_prob_per_variable = output_distribution.log_prob(x)
            return jnp.sum(log_prob_per_variable, where=mask)
