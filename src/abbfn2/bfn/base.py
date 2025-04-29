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

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import jax.numpy as jnp
from hydra.utils import instantiate
from jax import Array
from jax.random import PRNGKey
from omegaconf import DictConfig

from abbfn2.bfn.output_network.noise_schedules import NoiseSchedule
from abbfn2.bfn.types import OutputNetworkPrediction, Theta


class BFNBase(ABC):
    """A base Bayesain Flow Network class that will have specific implementations for different data modalities."""

    def __init__(self, **kwargs):
        """Initialise the BFN.

        This initialisation pattern is chosen to allow for the use of Hydra to configure the BFN whilst still
        having a DictConfig object for the configuration (by default Hydra instantiates passes the configuration
        as a series of keyword arguments).

        Args:
            output_network_cfg (Optional[DictConfig]): The configuration for the output network.  Default is None.
            **kwargs: The configuration for the BFN.
        """
        self.cfg = DictConfig(kwargs)

        if not isinstance(self.cfg.variables_shape, Iterable):
            self.cfg.variables_shape = (self.cfg.variables_shape,)
        else:
            self.cfg.variables_shape = tuple(self.cfg.variables_shape)

        if "output_network" not in self.cfg:
            self.cfg.output_network = None

    def init(self) -> Any:
        """Initialise the Bayesian Flow Network.

        Returns:
            params (Any): The learnable params of the BFN.
        """

        self.noise_schedule: NoiseSchedule = instantiate(self.cfg.noise_schedule)
        schedule_params = self.noise_schedule.init()

        params = {"noise_schedule": schedule_params}
        return params

    def compute_beta(self, params: Any, t: float) -> Array:
        """Compute the accuracy schedule at time t.

        β(t) is monotonically increasing in time such that β(0)=0 and dβ(t)/dt = α(t).

        Args:
           params (Any): The learnable params of the BFN (specifically, of the noise schedule).
           t (float): The time.

        Returns:
            Array: Per-variable accuracy schedule values (i.e. has shape with cfg.variables_shape).
        """
        if "noise_schedule" in params:
            params = params["noise_schedule"]
        t = jnp.full(self.cfg.variables_shape, fill_value=t)
        return self.noise_schedule.beta(t)

    def apply_output_network(
        self,
        params: Any,
        theta: Theta,
        t: float,
    ) -> OutputNetworkPrediction:
        """Apply the output network to compute parameters of the output distribution.

        Args:
            params (Any): The learnable params of the BFN (specifically, of the output network).
            key (PRNGKey): A random seed for the output network
            theta (Theta): Parameters of the input distribution.
            t (float): The time.
        Returns:
            OutputNetworkPrediction: Prediction of the output network.
        """
        beta = self.compute_beta(params, t)
        if "output_network" in params:
            params = params["output_network"]
        return self._apply_output_network_fn(params, theta, t, beta)

    @abstractmethod
    def get_prior_input_distribution(self) -> Theta:
        """Initialises the parameters of an uninformed input distribution."""
        pass

    @abstractmethod
    def sample_sender_distribution(
        self,
        x: Array,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a noise sample for the ground-truth (x) from the sender distribution.

        Specifically, this function samples y ~ p_S(y|x;α).

        Args:
            x (Array): The ground truth data.
            alpha (Array): A per-variable accuracy parameter.
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the sender distribution.
        """
        pass

    @abstractmethod
    def sample_receiver_distribution(
        self,
        pred: OutputNetworkPrediction,
        alpha: Array,
        key: PRNGKey,
    ) -> Array:
        """Generate a sample from the receiver distribution.

        Specifically, this function samples y ~ p_R(y|θ;t,α).  Note that the function takes
        as input pred (OutputNetworkPrediction) which is a function of θ;t, so we can rewrite
        this as y ~ p_R(y|pred(θ;t),α).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            alpha (Array): A per-variable accuracy parameter.
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the receiver distribution.
        """
        pass

    @abstractmethod
    def update_distribution(
        self,
        Theta: Array,
        y: Array,
        alpha: Array | None = None,
        conditional_score: Array | None = None,
        conditional_mask: Array | None = None,
    ) -> Theta:
        """Apply Bayesian update to distribution parameters given sample of receiver distribution.

        Args:
            Theta (Theta): Parameters of the distribution modelled by the BFN.
            y (Array): The sample from the sender distribution.
            alpha (Optional[Array]): The noise term for the sender distribution, per variable.
            conditional_score (Optional[Array]): Per-variable conditional score
                (gradient of log prob of conditional data wrt input parameters)
                used to update the input parameters for conditional SDE sampling.
                If None, update reverts to unconditional.
            conditional_mask (Optional[Array]): Per-variable mask for conditional sampling.
                The conditional update only happens where the mask is False
                (no need if the mask is True since the ground truth is already known).

        Returns:
            theta (Theta): Updated parameters of the distribution.
        """
        pass

    @abstractmethod
    def conditional_log_prob(
        self,
        pred: OutputNetworkPrediction,
        x: Array | None,
        mask: Array | None,
        theta: Theta,
    ) -> float:
        """Calculate the log probability of data x given input parameters theta (used to determine the conditional score for SDE sampling).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            x (Optional[Theta]): Conditioning data.
            mask: (Optional[Theta]): Per-variable boolean mask for the conditioning data.
                    Valid masks can be broadcast the shape of x and are True (False) if a conditional variable is used (unused).
            theta (Optional[Theta]): Parameters of the input distribution.


        Returns:
            The summed log prob over all the variables in x where mask=True. Returns 0 if x is None
        """
        pass
