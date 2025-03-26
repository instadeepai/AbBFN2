from dataclasses import dataclass
from typing import Any

import jax
from distrax import Categorical, Normal
from flax import struct
from jax import Array
from jax import numpy as jnp
from jax.lax import scan
from jax.random import PRNGKey

from abbfn2.bfn import BFN, ContinuousBFN, DiscreteBFN, MultimodalBFN
from abbfn2.bfn.types import (
    OutputNetworkPrediction,
    OutputNetworkPredictionMM,
    Theta,
    ThetaContinuous,
    ThetaDiscrete,
    ThetaMM,
)
from abbfn2.sample.functions.base import BaseSampleFn


def _twisted_importance_logit_cts_mean(
    mean: Array,
    y: Array,
    theta_old: ThetaContinuous,
    theta_new: ThetaContinuous,
    alpha: Array,
    mask: Array,
) -> float:
    receiver_distribution = Normal(mean, 1 / jnp.sqrt(alpha))
    proposal_log_prob = jnp.sum(jnp.where(mask, 0, receiver_distribution.log_prob(y)))
    y_unconditional = (
        (theta_new.mu * theta_new.rho) - (theta_old.mu * theta_old.rho)
    ) / alpha
    prior_log_prob = jnp.sum(
        jnp.where(mask, 0, receiver_distribution.log_prob(y_unconditional))
    )
    return prior_log_prob - proposal_log_prob


def get_twisted_importance_logit_continuous_dm(
    bfn: ContinuousBFN,
    pred: OutputNetworkPrediction,
    y: Array,
    theta_old: ThetaContinuous,
    theta_new: ThetaContinuous,
    alpha: Array,
    mask: Array,
) -> float:
    """Calculate the importance logit for twisted SMC sampling with a continuous BFN.

        Given by log prob of receiver sample y under the unconditional prior minus
        log prob of y under the conditional (twisted) proposal distribution

    Args:
        bfn (ContinuousBFN): The BFN model.
        pred (OutputNetworkPredictionContinuous): Prediction of the output network.
        y (Array): The receiver/sender sample (per variable).
        theta_old (ThetaContinuous): The previous input distribution parameters (per variable)
        theta_new (ThetaContinuous): The updated input distribution parameters (per variable)
        alpha (Array): A per-variable accuracy parameter.
        mask (Array): Variable-wise mask indicating whether the ground-truth is known (False means the network does not know the ground-truth).

    Returns:
        float: The particle logit.
    """
    mean = pred.x
    return _twisted_importance_logit_cts_mean(
        mean,
        y,
        theta_old,
        theta_new,
        alpha,
        mask,
    )


def get_twisted_importance_logit_discrete_dm(
    bfn: DiscreteBFN,
    pred: OutputNetworkPrediction,
    y: Array,
    theta_old: ThetaDiscrete,
    theta_new: ThetaDiscrete,
    alpha: Array,
    mask: Array,
) -> float:
    """Calculate the importance logit for twisted SMC sampling with a discrete BFN.

        Logit is the log prob of the receiver sample y under the unconditional prior minus
        the log prob of y under the conditional (twisted) proposal distribution

    Args:
        bfn (DiscreteBFN): The BFN model.
        pred (OutputNetworkPredictionContinuous): Prediction of the output network.
        y (Array): The receiver/sender sample (per variable).
        theta_old (ThetaContinuous): The previous input distribution parameters (per variable)
        theta_new (ThetaContinuous): The updated input distribution parameters (per variable)
        alpha (Array): A per-variable accuracy parameter.
        mask (Array): Variable-wise mask indicating whether the ground-truth is known (False means the network does not know the ground-truth).

    Returns:
        float: The particle logit.
    """
    probs = pred.to_distribution().probs
    K = probs.shape[-1]
    mu = alpha[..., None] * (K * probs - 1)
    sigma = jnp.sqrt(alpha[..., None] * K * jnp.ones(K))
    receiver_distribution = Normal(mu, sigma)
    proposal_log_prob = jnp.sum(
        jnp.where(mask[..., None], 0, receiver_distribution.log_prob(y))
    )
    y_unconditional = theta_new.logits - theta_old.logits
    prior_log_prob = jnp.sum(
        jnp.where(mask[..., None], 0, receiver_distribution.log_prob(y_unconditional))
    )
    return prior_log_prob - proposal_log_prob


def get_twisted_particle_logit(
    bfn: MultimodalBFN,
    pred: OutputNetworkPredictionMM,
    alpha: dict[str, Array],
    y: dict[str, Array],
    theta_old: ThetaMM,
    theta_new: ThetaMM,
    cond_log_p_old: dict[str, float],
    cond_log_p_new: dict[str, float],
    mask: dict[str, Array],
) -> float:
    """Calculate the log particle weights for a twisted SMC particle using a multimodal BFN model.
    Adapted to BFNs from Algorithm 1. in "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models", Wu et. al.

    The function computes logits for each data-mode independently and sums them to determine the particle's likelihood of being re-sampled.
    Higher logit scores correspond to particles whose samples are more consistent with the ground truth.

    Concretely, denoting theta_old / theta_new as θ_0 / θ_1, the conditioning data as x and twisted/untwisted probabilities
    as p' and p respectively, in Wu et al. the formula for the particle weight is given by:

        w = [ p(θ_1|θ_0) / p'(θ_1|θ_0,x) ] x [ p'(x|θ_1) / p'(x|θ_0) ]

    Now, recall that a parameter step in a (score-guided) BFN is done by following:

        1. Given θ_0, make the network prediction (pred) which defines the output distribution p_o(.|θ_0)).  Also
           compute the score of the conditioning data under this prediction (s(θ_0, x)).
        2. From the output distribution, obtain and sample the receiver distribution; y ~ p_r(y|θ_0,α).
        3. Update the input distribution to θ_1 using the sampled receiver distribution and the score;
           θ_1 = update(θ_0, y, s(θ_0, x)).

    This means that the probability of parameters θ_1 given θ_0 is the probability of the sample, y, under the
    receiver distribution.  Therefore, if we have y and y_uncond s.t.
        θ_1 = update(θ_0, y, s(θ_0, x));  θ_1 = update(θ_0, y_uncond, None)
    then we can define p(θ_1|θ_0) = p_r(y_uncond|θ_0,α) and p'(θ_1|θ_0) = p_r(y|θ_0,α).

    Therefore, we can rewrite the weight as:

            w = [ p_r(y_uncond|θ_0,α) / p_r(y|θ_0,α) ] x [ p'(x|θ_1) / p'(x|θ_0) ].

    Or, in terms of log probabilities:

           log(w) = [ log p_r(y_uncond|θ_0,α) - log p_r(y|θ_0,α) ] + [ log p'(x|θ_1) - log p'(x|θ_0) ].

    Args:
        bfn (MultimodalBFN): The BFN model.
        pred (OutputNetworkPredictionMM): Prediction of the output network at t (i.e. pred(θ_0)).
        y (dict[str, Array]): The receiver/sender sample (i.e. y ~ p_r(y|θ_0,α)).
        alpha (dict[str, Array]): The receiver/sender accuracy.
        theta_old (ThetaMM): The previous input distribution parameters (θ_0).
        theta_new (ThetaMM): The updated input distribution parameters (θ_1).
        cond_log_p_old (dict[str, float]): the log probability of the conditioning data at the previous sampling step (log p'(x|θ_0)).
        cond_log_p_new (dict[str, float]): the log probability of the conditioning data at the current sampling step (log p'(x|θ_1))
        mask (dict[str, Array]): Variable-wise mask indicating whether the ground-truth is known (False means the network does not know the ground-truth).

    Returns:
        float: The log particle weight
    """

    def get_twisted_importance_logit_for_dm(
        bfn: BFN,
        pred: OutputNetworkPrediction,
        y: Array,
        theta_old: Theta,
        theta_new: Theta,
        alpha: Array,
        mask: Array,
    ):
        if isinstance(bfn, ContinuousBFN):
            logit = get_twisted_importance_logit_continuous_dm(
                bfn, pred, y, theta_old, theta_new, alpha, mask
            )
        elif isinstance(bfn, DiscreteBFN):
            logit = get_twisted_importance_logit_discrete_dm(
                bfn, pred, y, theta_old, theta_new, alpha, mask
            )
        else:
            raise ValueError(f"Unsupported BFN type: {type(bfn)}")

        return logit

    importance_logits_per_dm = jax.tree_map(
        get_twisted_importance_logit_for_dm,
        bfn.bfns,
        pred,
        y,
        theta_old,
        theta_new,
        alpha,
        mask,
    )
    importance_logit = sum(importance_logits_per_dm.values())
    logit = importance_logit + cond_log_p_new - cond_log_p_old
    return logit


@struct.dataclass
class SampleState:
    """State for the SampleFn.

    Attributes:
        t (float): The time at which the prediction was made.
        theta (ThetaMM): The input distribution parameters used at time t.
        pred (OutputNetworkPredictionMM): The network prediction at time t.
        log_prob (float): The log probability of the conditioning data using the network prediction at time t.
        old_log_prob (float): The log probability of the conditioning data using the network prediction at the previous time step (e.g. t-dt).
        score (float): The score of the conditioning data using the network prediction at time t.
        particle_logit (float): The log weight of this particle.
    """

    t: float
    theta: ThetaMM
    pred: OutputNetworkPredictionMM
    log_prob: float
    old_log_prob: float
    score: float
    particle_logit: float


@dataclass
class TwistedSDESampleFn(BaseSampleFn):
    """Creates the Twisted SDE sampling function for the BFN model.

    Adapted to BFNs from "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models", Wu et al.

    Args:
        bfn (BFN): The BFN model.
        num_steps (int): The number of steps to iterate for generating samples.
        time_schedule (TimeScheduleFn): The time schedule function.
        greedy (bool): Whether to sample the mode of the distribution (greedy) or sample from the distribution. Defaults to True.
        use_self_conditioning (bool): Whether to condition on the previous prediction when generating the next prediction. Defaults to False.
        num_particles (int): The number of particles to use in the SMC algorithm.
        max_score (float | None): Range at which to clip the conditional score. Defaults to 1.0.
            Set to None for no clipping (can lead to numerical instability)
        twist_scale (float | None): Scale used to multiply the conditional log probs for the twisting function.
            Defaults to None (no multiplication)
            If > 1.0, should bias sampling to make the conditioning data more probable, at the expense of reduced diversity.
            Similar to classifier scale in classifier guidance.
        mask_receiver_sample (bool): Whether to mask the receiver with the sender sample for the conditioning data. Defaults to True.
    """

    num_particles: int = 1
    max_score: float | None = 1.0
    twist_scale: float | None = None
    mask_receiver_sample: bool = True

    def _get_score_and_pred(
        self,
        key: PRNGKey,
        theta: ThetaMM,
        t: float,
        params: Any,
        x: dict[str, Array],
        mask_sample: dict[str, Array],
        pred_cond: OutputNetworkPredictionMM | None = None,
        t_cond: float | None = None,
    ):
        """Get the conditional score, log prob of the conditioning data, and prediction from the network.

        The score is the gradient of the log probability of the conditioning data (x) at the network output with
        respect to the network input (theta).  This function also clips this score and `twists` (i.e. scales)
        the log prob if needed.

        Args:
            key (PRNGKey): The random key used for generating samples.
            theta (ThetaMM): The input distribution.
            t (float): The time at which to condition the network.
            params (Any): The network parameters.
            x (Dict[str, Array]): The "ground-truth" sample on which to condition generation.
            mask_sample (Dict[str, Array]): Variable-wise mask determining which regions of the ground-truth is visible during sampling.
            pred_cond (OutputNetworkPredictionMM | None): The previous prediction to condition on. Defaults to None.
            t_cond (float | None): The time at which the previous prediction was made. Defaults to None.

        Returns:
            Tuple[float, float, OutputNetworkPredictionMM]: The conditional score, log prob and prediction.
        """

        def conditional_log_prob_fn(
            theta: ThetaMM,
        ) -> tuple[float, OutputNetworkPredictionMM]:
            pred = self.bfn.apply_output_network(
                params,
                key,
                theta,
                t,
                mask=None,
                pred_cond=pred_cond if self.use_self_conditioning else None,
                t_cond=t_cond if self.use_self_conditioning else None,
            )
            log_prob = self.bfn.conditional_log_prob(pred, x, mask_sample, theta)
            return log_prob, pred

        (log_prob, pred), grad_log_prob = jax.value_and_grad(
            conditional_log_prob_fn, has_aux=True
        )(theta)

        # Clip the score if needed.
        cond_score = grad_log_prob
        if self.max_score is not None:
            cond_score = jax.tree_map(
                lambda s: jnp.clip(s, -self.max_score, self.max_score),
                cond_score,
            )

        # "Twist" (scale) the log_prob if needed.
        if self.twist_scale is not None:
            log_prob *= self.twist_scale

        return cond_score, log_prob, pred

    def _get_sample(
        self,
        key: PRNGKey,
        pred: OutputNetworkPredictionMM,
        x: dict[str, Array],
        mask_sample: dict[str, Array],
        alpha: dict[str, Array],
    ):
        """Sample from the sender and receiver distributions and choose between them based on the mask.

        The process is as follows:
            1. Bob "hallucinates" Alice without knowing the ground truth.
            2. Alice sends the (noised) ground truth.
            3. Choose between samples from receiver and sender distibutions depending on the mask.

        Args:
            key (PRNGKey): The random key used for generating samples.
            pred (OutputNetworkPredictionMM): The network prediction.
            x (Dict[str, Array]): The "ground-truth" sample on which to condition generation.
            mask_sample (Dict[str, Array]): Variable-wise mask determining which regions of the ground-truth is visible during sampling.
            alpha (Dict[str, Array]): Per-variable accuracy schedule values for each data mode.

        Returns:
            Dict[str, Array]: The sampled values.
        """
        key_sender, key_receiver = jax.random.split(key, 2)
        y = self.bfn.sample_receiver_distribution(pred, alpha, key_receiver)

        if self.mask_receiver_sample:
            y_sen = self.bfn.sample_sender_distribution(x, alpha, key_sender)
            # We add ending dimensions to the mask such that is can be broadcast over y_sen, y_rec
            y = jax.tree_util.tree_map(
                lambda y_sen, y_rec, m: jnp.where(
                    m.reshape(m.shape + (1,) * (y_sen.ndim - m.ndim)),
                    y_sen,
                    y_rec,
                ),
                y_sen,
                y,
                mask_sample,
            )

        return y

    def __call__(
        self,
        key: PRNGKey,
        params: dict,
        x: dict[str, Array],
        mask_sample: dict[str, Array],
    ) -> Array:
        """Run the Twisted SDE sampling algorithm.

        Parameters:
            key (PRNGKey): The random key used for generating samples.
            params (dict): The network params.
            x (dict[str, Array]): The "ground-truth" sample on which to condition generation.
            mask_sample (dict[str, Array]): Variable-wise mask determining which regions of the ground-truth
                is visible during sampling. i.e. 0 means the network does not know the ground-truth.

        Returns:
            Array: The generated samples.
        """
        # Prepare model input.
        theta = self.bfn.get_prior_input_distribution()
        for dm in theta.keys():
            if mask_sample[dm] is None:
                mask_sample[dm] = jnp.ones_like(x[dm])

        # Run network and compute score at t=0.
        # Note that (see line 3, Alg 1 of Wu et al.) the initial weight is log_prob(x | theta_0).  However,
        # as we are replicating the state across particles, we can set the initial weight to 0.  Note that if
        # the network is not-deterministic, we should also vmap this step and use the correct weights.
        key, key_output = jax.random.split(key)
        cond_score, log_prob, pred = self._get_score_and_pred(
            key_output, theta, t=0.0, params=params, x=x, mask_sample=mask_sample
        )
        sample_state = SampleState(
            t=0.0,
            theta=theta,
            pred=pred,
            log_prob=log_prob,
            old_log_prob=0.0,
            score=cond_score,
            particle_logit=0.0,
        )

        # Replicate the state across num_particles.
        sample_states = jax.tree_util.tree_map(
            lambda x: jax.lax.broadcast(x, [self.num_particles]),
            sample_state,
        )

        def loop_body(states: SampleState, xs: tuple[int, PRNGKey]):
            """Loop body for a single particle.

            Args:
                state (SampleState): The current state.
                xs (tuple[int, PRNGKey]): The loop variables (i, key) where i is the current step and key is the random key.

            Returns:
                SampleState: The updated state.
            """
            i, key = xs
            t_start, t_end = self.time_schedule(i, self.num_steps)

            # Compute noise for t to t + dt.
            beta = self.bfn.compute_beta(params, t_start)
            beta_next = self.bfn.compute_beta(params, t_end)
            alpha = jax.tree_util.tree_map(lambda b2, b1: b2 - b1, beta_next, beta)

            # Resample particles with weights from previous step.
            particle_idxs = Categorical(logits=states.particle_logit).sample(
                seed=key,
                sample_shape=self.num_particles,
            )
            states = jax.tree_util.tree_map(lambda arr: arr[particle_idxs], states)

            def process_particle(state: SampleState, key):
                key_sample, key_output = jax.random.split(key, 2)

                # Update theta from t to t + dt using the network prediction at t.
                y = self._get_sample(key_sample, state.pred, x, mask_sample, alpha)
                theta = self.bfn.update_distribution(
                    state.theta, y, alpha, state.score, mask_sample
                )

                # Calculate the particle weights at t + dt
                # TB comment: I believe this implementation will match what we previously used.
                # Note: having the particle logit here means we get particle weights but then forward
                #       pass the network before resampling.  This is fine so long as the network is deterministic...
                # particle_logit = get_twisted_particle_logit(
                #     bfn=self.bfn,
                #     pred=state.pred,  # pred(θ_0) to obtain p_r(y|θ_0,α)
                #     alpha=alpha,  # α
                #     y=y,  # y ~ p_r(y|θ_0,α)
                #     theta_old=state.theta,  # θ_0
                #     theta_new=theta,  # θ_1
                #     cond_log_p_old=state.old_log_prob,  # log p'(x|θ_0) (or log p'(x|θ_-1)???)
                #     cond_log_p_new=state.log_prob,  # log p'(x|θ_1) (or log p'(x|θ_0)???)
                #     mask=mask_sample,  # mask
                # )

                # Run the network at t + dt.
                cond_score, log_prob, pred = self._get_score_and_pred(
                    key_output,
                    theta,
                    t=t_end,
                    params=params,
                    x=x,
                    mask_sample=mask_sample,
                    pred_cond=state.pred,
                    t_cond=state.t,
                )

                # Calculate the particle weights at t + dt
                # TB comment: This implementation is what I got to from Wu et al.
                particle_logit = get_twisted_particle_logit(
                    bfn=self.bfn,
                    pred=state.pred,  # pred(θ_0) to obtain p_r(y|θ_0,α)
                    alpha=alpha,  # α
                    y=y,  # y ~ p_r(y|θ_0,α)
                    theta_old=state.theta,  # θ_0
                    theta_new=theta,  # θ_1
                    cond_log_p_old=state.log_prob,  # log p'(x|θ_0)
                    cond_log_p_new=log_prob,  # log p'(x|θ_1)
                    mask=mask_sample,  # mask
                )

                state = SampleState(
                    t=t_end,
                    theta=theta,
                    pred=pred,
                    log_prob=log_prob,
                    old_log_prob=state.log_prob,
                    score=cond_score,
                    particle_logit=particle_logit,
                )
                return state, y

            states, ys = jax.vmap(process_particle)(
                states,
                jax.random.split(key, self.num_particles),
            )

            return states, ys

        # Run sampling for t_start, t_end in = [(0,1/N),(1/N, 2/N), ..., (1-1/N, 1)].
        key, ks = jax.random.split(key)
        sample_states, ys = scan(
            loop_body,
            sample_states,
            (jnp.arange(self.num_steps), jax.random.split(ks, self.num_steps)),
        )

        samples = jax.vmap(self._sample_from_network_prediction)(
            jax.random.split(key, self.num_particles),
            sample_states.pred,
        )
        return samples