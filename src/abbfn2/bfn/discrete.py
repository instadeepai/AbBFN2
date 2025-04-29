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

import jax
import jax.numpy as jnp
from distrax import Categorical, Normal
from jax import Array
from jax.random import PRNGKey

from abbfn2.bfn.base import BFNBase
from abbfn2.bfn.types import OutputNetworkPredictionDiscrete, ThetaDiscrete


class DiscreteBFN(BFNBase):
    """Discrete-variable Bayesian Flow Network."""

    def get_prior_input_distribution(self) -> ThetaDiscrete:
        """Initialises the parameters of an uninformed input distribution.

        Returns:
            logits (ThetaDiscrete): Uniform prior distribution parameters.
        """
        logits = jnp.zeros(self.cfg.variables_shape + (self.cfg.num_classes,))
        return ThetaDiscrete(logits=logits)

    def sample_sender_distribution(
        self,
        x: Array,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a noise sample for the ground-truth (x) from the sender distribution.

        Specifically, this function samples y ~ p_S(y|x;α) = N(y|α(K e_x − 1), αKI).

        Args:
            x (Array): The ground truth data; sample from K-class categorical distribution over the variables (i.e. shape [...var_shape...]).
            alpha (Array): A per-variable accuracy parameter (shape [...var_shape...]).
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the sender distribution.

        Notes:
            This function implementes eq. 157 in Graves et al.
        """
        K = self.cfg.num_classes

        # mu and sigma have shapes: (...var_shape..., K)
        mu = alpha[..., None] * (K * jax.nn.one_hot(x, num_classes=K, axis=-1) - 1)
        sigma = jnp.sqrt(alpha[..., None] * K * jnp.ones(K))

        # This is the distribution over the logits of the sender distribution,
        # built explicitly to support the re-parameterisation trick
        dist_z = Normal(jnp.zeros_like(mu), jnp.ones_like(sigma))
        z = dist_z.sample(seed=key)
        y = mu + sigma * z  # Note: Normal has scale = s.d. hence sqrt.

        return y

    def sample_receiver_distribution(
        self,
        pred: OutputNetworkPredictionDiscrete,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a sample from the receiver distribution.

        Specifically, this function samples y ~ p_R(y|θ;t,α).  Note that the function takes
        as input pred (OutputNetworkPrediction) which is a function of θ;t, so we can rewrite
        this as y ~ p_R(y|pred(θ;t),α).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            alpha (Array): A per-variable accuracy parameter (shape [...var_shape...]).
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the receiver distribution.

        Notes:
            This function implements eq. 158-159 in Graves et al.
        """
        probs = Categorical(logits=pred.logits).probs
        K = self.cfg.num_classes

        # alpha: [...var_shape...], mu: [...var_shape..., K], sigma: [...var_shape..., K]
        mu = alpha[..., None] * (K * probs - 1)
        sigma = jnp.sqrt(alpha[..., None] * K * jnp.ones(K))

        # This is the distribution over the logits of the sender distribution,
        # built explicitly to support the re-parameterisation trick
        dist_z = Normal(jnp.zeros_like(mu), jnp.ones_like(sigma))
        z = dist_z.sample(seed=key)
        y = mu + sigma * z

        return y

    def update_distribution(
        self,
        theta: ThetaDiscrete,
        y: Array,
        alpha: Array | None = None,
        conditional_score: Array | None = None,
        conditional_mask: Array | None = None,
    ) -> ThetaDiscrete:
        """Apply update to distribution parameters given sample of receiver distribution.

        Args:
            theta (ThetaDiscrete): Parameters of the output distribution.
            y (Array): The sample from the sender distribution.
            alpha (Optional[Array]): The noise term for the sender distribution, per variable, with length D.
            conditional_score (Optional[Array]): Per-variable conditional score
                (gradient of log prob of conditional data wrt input parameters)
                used to update the input parameters for conditional SDE sampling.
                If None, update reverts to unconditional.
            conditional_mask (Optional[Array]): Per-variable mask for conditional sampling.
                The conditional update only happens where the mask is False
                (no need if the mask is True since the ground truth is already known).

        Returns:
            theta (ThetaDiscrete): Updated parameters of the distribution.
        """
        logits = theta.logits + y
        if conditional_score is not None:
            num_classes = logits.shape[-1]
            logit_delta = alpha[..., None] * num_classes * conditional_score.logits
            if conditional_mask is not None:
                logit_delta = jnp.where(conditional_mask[..., None], 0, logit_delta)
            logits += logit_delta
        return ThetaDiscrete(logits=logits)

    def conditional_log_prob(
        self,
        pred: OutputNetworkPredictionDiscrete,
        x: Array | None,
        mask: Array | None,
        theta: ThetaDiscrete,
    ) -> float:
        """Calculate log p(x|theta) (used to determine the conditional score for SDE sampling).

        Args:
            pred (OutputNetworkPredictionDiscrete): Prediction of the output network.
            x (Optional[Array]): Conditioning data.
            mask (Optional[Array]): Per-variable boolean mask for the conditioning data.
                    Valid masks can be broadcast the shape of x and are True (False) if a conditional variable is used (unused).
            theta (ThetaDiscrete): Parameters of the input distribution.
            conditional_score (Optional[Array]): Per-variable conditional score
                (gradient of log prob of conditional data wrt input parameters)
                used to update the input parameters for conditional SDE sampling.
                If None, update reverts to unconditional.
            conditional_mask (Optional[Array]): Per-variable mask for conditional sampling.
                The conditional update only happens where the mask is False
                (no need if the mask is True since the ground truth is already known).


        Returns:
            The summed log prob over all the variables in x where mask=True. Returns 0 if x is None
        """
        if x is None:
            return 0
        else:
            log_prob_per_variable = pred.to_distribution().log_prob(x)
            return jnp.sum(log_prob_per_variable, where=mask)
