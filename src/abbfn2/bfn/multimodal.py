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

from typing import Any

import jax
from jax import Array
from jax.random import PRNGKey, split
from omegaconf import DictConfig

from abbfn2.bfn.base import BFNBase
from abbfn2.bfn.output_network.model import BFNMultimodalOutput
from abbfn2.bfn.types import OutputNetworkPredictionMM, ThetaMM


class MultimodalBFN:
    """A multi-modal Bayesian Flow Network."""

    def __init__(
        self,
        bfns: dict[str, BFNBase],
        output_network_cfg: DictConfig | None = None,
    ):
        """Initialise the multi-modal Bayesian Flow Network.

        Args:
            bfns (Dict[str, BFNBase]): A dictionary of BFNs, one for each data mode.
            output_network_cfg (Optional[DictConfig]): The configuration for the output network.
        """
        self.bfns = bfns
        self.data_modes = sorted(list(self.bfns.keys()))  # noqa
        self.output_network_cfg = output_network_cfg
        self._split_key_for_dms = lambda key: {  # noqa
            dm: k for dm, k in zip(self.data_modes, split(key, len(self.data_modes)))
        }
        self._replicate_across_dms = lambda x: {dm: x for dm in self.data_modes}

    def init(self, key: PRNGKey) -> Any:
        """Initialise the Bayesian Flow Network.

        Args:
            key (PRNGKey): The key to use for initialising the BFN

        Returns:
            params (Any): The learnable params of the BFN
        """

        # Initialise the noise schedule for each BFN.
        noise_schedule_params = {}
        for dm, bfn in self.bfns.items():
            noise_schedule_params[dm] = bfn.init()["noise_schedule"]

        # Build multimodal output network.
        output_network_fn = BFNMultimodalOutput(
            {dm: bfn.cfg for dm, bfn in self.bfns.items()},
            self.output_network_cfg,
        )

        theta = self.get_prior_input_distribution()
        t = 0.5
        beta = self.compute_beta(noise_schedule_params, t)

        output_network_params = output_network_fn.init(
            key,
            theta,
            t,
            beta,
        )
        self._apply_output_network_fn = output_network_fn.apply

        return {
            "output_network": output_network_params,
            "noise_schedule": noise_schedule_params,
        }

    def get_prior_input_distribution(self) -> ThetaMM:
        """Initialises the parameters of an uninformed input distribution.

        Returns:
            ThetaMM: Dictionary of input distributions.
        """
        return jax.tree_util.tree_map(
            lambda bfn: bfn.get_prior_input_distribution(),
            self.bfns,
        )

    def apply_output_network(
        self,
        params: dict[str, Any],
        theta: ThetaMM,
        t: float,
    ) -> OutputNetworkPredictionMM:
        """Apply the output network to compute parameters of the output distribution.

        Args:
            params (Any): The learnable params of the BFN (specifically, of the output network).
            key (PRNGKey): A random seed for the output network
            theta (ThetaMM): Parameters of the input distribution.
            t (float): The time.
        Returns:
            OutputNetworkPredictionMM: Prediction of the output network.
        """
        beta = self.compute_beta(params, t)
        if "output_network" in params:
            params = params["output_network"]
        pred = self._apply_output_network_fn(params, theta, t, beta)
        return pred

    def compute_beta(self, params: Any, t: float) -> Array:
        """Compute the accuracy schedule at time t.

        β(t) is monotonically increasing in time such that β(0)=0 and dβ(t)/dt = α(t).

        Args:
           params (Any): The learnable params of the BFN (specifically, of the noise schedule).
           t (float): The time.

        Returns:
            Array: Per-variable accuracy schedule values (i.e. with shape [N]).
        """
        if "noise_schedule" in params:
            params = params["noise_schedule"]
        return jax.tree_util.tree_map(
            lambda bfn, params: bfn.compute_beta(params, t),
            self.bfns,
            params,
        )

    def sample_sender_distribution(
        self,
        x: dict[str, Array | None],
        alpha: dict[str, Array],
        key: PRNGKey,
    ) -> dict[str, Array | None]:
        """Generate a noise sample for the ground-truth (x) from the sender distribution.

        Args:
            x (Dict[str, Array]): The ground truth data (shape [N,...]).
            alpha (Dict[str, Array]): A per-variable accuracy parameter (shape [N]).
            key: PRNGKey for sampling.

        Returns:
            y (Dict[str, Array]): The sample from the sender distribution.
        """
        keys = self._split_key_for_dms(key)
        return jax.tree_util.tree_map(
            lambda bfn, x, alpha, key: (
                None if x is None else bfn.sample_sender_distribution(x, alpha, key)
            ),
            self.bfns,
            x,
            alpha,
            keys,
        )

    def sample_receiver_distribution(
        self,
        pred: OutputNetworkPredictionMM,
        alpha: dict[str, Array],
        key: PRNGKey,
    ) -> dict[str, Array]:
        """Generate a sample from the receiver distribution (output distribution convolved with the sender distribution).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            alpha (Array): A per-variable accuracy parameter (shape [N]).
            key: PRNGKey for sampling.

        Returns:
            y (Array): The sample from the sender distribution.
        """
        keys = self._split_key_for_dms(key)
        return jax.tree_util.tree_map(
            lambda bfn, pred, alpha, key: bfn.sample_receiver_distribution(
                pred,
                alpha,
                key,
            ),
            self.bfns,
            pred,
            alpha,
            keys,
        )

    def update_distribution(
        self,
        theta: ThetaMM,
        y: dict[str, Array],
        alpha: dict[str, Array | None],
        conditional_score: dict[str, Array | None] | None = None,
        conditional_mask: dict[str, Array | None] | None = None,
    ) -> ThetaMM:
        """Apply Bayesian update to distribution parameters given sample of receiver distribution.

        Args:
            theta (ThetaMM): Parameters of the distribution modelled by the BFN.
            y (Dict[str, Array]): The sample from the sender distribution.
            alpha (Dict[str, Union[Array, None]]): The noise term for the sender distribution, per variable, with length D.
            conditional_score (dict[str, Array | None] or None): Per modality, Per-variable conditional score
                (gradient of log prob of conditional data wrt input parameters)
                used to update the input parameters for conditional SDE sampling.
                If None, update reverts to unconditional.
            conditional_mask (dict[str, Array | None] or None): Per modality, Per-variable mask for conditional sampling.
                The conditional update only happens where the mask is False
                (no need if the mask is True since the ground truth is already known).

        Returns:
            theta (ThetaMM): Updated parameters of the distribution.
        """
        if conditional_score is None:
            conditional_score = self._replicate_across_dms(conditional_score)
        if conditional_mask is None:
            conditional_mask = self._replicate_across_dms(conditional_mask)
        return jax.tree_util.tree_map(
            lambda bfn, theta, y, alpha, conditional_score, conditional_mask: bfn.update_distribution(
                theta, y, alpha, conditional_score, conditional_mask
            ),
            self.bfns,
            theta,
            y,
            alpha,
            conditional_score,
            conditional_mask,
        )

    def conditional_log_prob(
        self,
        pred: OutputNetworkPredictionMM,
        x: dict[str, Array | None],
        mask: dict[str, Array | None],
        theta: ThetaMM,
    ) -> float:
        """Calculate log p(x|theta) (used to determine the conditional score for SDE sampling).

        Args:
            pred (OutputNetworkPrediction): Prediction of the output network.
            x (dict[str, Array | None]): Conditioning data.
            mask: (dict[str, Array | None]): Per-variable boolean mask for the conditioning data.
                    Valid masks can be broadcast the shape of x and are True (False) if a conditional variable is used (unused).
            theta (theta): Parameters of the input distribution.

        Returns:
            The summed log prob over all the variables in x where mask=True
        """
        log_probs = jax.tree_util.tree_map(
            lambda bfn, pred, x, mask, theta: bfn.conditional_log_prob(
                pred, x, mask, theta
            ),
            self.bfns,
            pred,
            x,
            mask,
            theta,
        )
        return sum(log_probs.values())
