import logging
from dataclasses import dataclass

import jax
from flax import struct
from jax import Array
from jax import numpy as jnp
from jax.lax import scan
from jax.random import PRNGKey

from abbfn2.bfn.types import OutputNetworkPredictionMM, ThetaMM
from abbfn2.sample.functions.base import BaseSampleFn


@struct.dataclass
class SampleState:
    """State for the SampleFn.

    Attributes:
        t (float): The time at which the prediction was made.
        theta (ThetaMM): The input distribution parameters used at time t.
        pred (OutputNetworkPredictionMM): The network prediction at time t.
    """

    t: float
    theta: ThetaMM
    pred: OutputNetworkPredictionMM


@dataclass
class SampleFn(BaseSampleFn):
    """Get the sampling function for the BFN model.

    This is the standard BFN sampling function introduced by Graves et al. (2023).  It is only adapted to unconditional
    sampling and does not support conditioning on ground-truth data or masks.

    Args:
        bfn (BFN): The BFN model.
        num_steps (int): The number of steps to iterate for generating samples.
        time_schedule (TimeScheduleFn): The time schedule function.
        greedy (bool): Whether to sample the mode of the distribution (greedy) or sample from the distribution.
        use_self_conditioning (bool): Whether to condition on the previous prediction when generating the next prediction. Defaults to False.
    """

    def __call__(
        self,
        key: PRNGKey,
        params: dict,
        x: dict[str, Array] | None = None,
        mask_sample: dict[str, Array] | None = None,
    ) -> jnp.array:
        """Generate samples by "hallucinating" Alice.

        Parameters:
            key (PRNGKey): The random key used for generating samples.
            params (dict): The network params.
            x (Dict[str, Array] | None): The "ground-truth" sample on which to condition generation.  Ignored in this function.
            mask_sample (Dict[str, Array] | None): Variable-wise mask determining which regions of the ground-truth. Ignored in this function.

        Returns:
            Array: The generated samples.

        Note:
            This function implements algorithm's 3, 6 and 9 from Graves et al.

        Raises:
            logging.warning: If conditioning on ground-truth data or mask is attempted.
        """
        if x is not None or mask_sample is not None:
            raise logging.warning(
                "Conditioning on ground-truth data or mask is not supported by SampleFn.  This conditioning information will be ignored."
            )

        # Prepare model input.
        theta = self.bfn.get_prior_input_distribution()

        # Run network at t=0
        key, key_output = jax.random.split(key)
        pred = self.bfn.apply_output_network(
            params,
            key_output,
            theta,
            t=0,
            mask=None,
            pred_cond=None,
            t_cond=None,
        )
        sample_state = SampleState(t=0.0, theta=theta, pred=pred)

        def loop_body(
            state: SampleState, xs: tuple[int, PRNGKey]
        ) -> tuple[SampleState, dict[str, Array]]:
            """Loop body for sampling.

            Args:
                state (SampleState): The current state.
                xs (tuple[int, PRNGKey]): The loop variables (i, key) where i is the current step and key is the random key.

            Returns:
                tuple[SampleState, dict[str, Array]]: The updated state and the sampled values.
            """
            i, key = xs
            key_output, key_receiver = jax.random.split(key, 2)
            t_start, t_end = self.time_schedule(i, self.num_steps)

            beta = self.bfn.compute_beta(params, t_start)
            beta_next = self.bfn.compute_beta(params, t_end)
            alpha = jax.tree_util.tree_map(lambda b2, b1: b2 - b1, beta_next, beta)

            # Update theta from t to t + dt using the network prediction at t.
            y = self.bfn.sample_receiver_distribution(state.pred, alpha, key_receiver)
            theta = self.bfn.update_distribution(state.theta, y, alpha)

            # Run the network at t + dt.
            pred = self.bfn.apply_output_network(
                params,
                key_output,
                theta,
                t_end,
                mask=None,
                pred_cond=state.pred if self.use_self_conditioning else None,
                t_cond=state.t if self.use_self_conditioning else None,
            )
            state = SampleState(t=t_end, theta=theta, pred=pred)

            return state, y

        # Run sampling for t_start, t_end in = [(0,1/N),(1/N, 2/N), ..., (1-1/N, 1)].
        key, ks = jax.random.split(key)

        sample_state, ys = scan(
            loop_body,
            sample_state,
            (jnp.arange(self.num_steps), jax.random.split(ks, self.num_steps)),
        )

        # Sample from the output network prediction.
        sample = self._sample_from_network_prediction(key, sample_state.pred)
    
        return sample
