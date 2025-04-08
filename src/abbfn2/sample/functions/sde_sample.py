from dataclasses import dataclass
from functools import partial

import jax
from jax import Array
from jax import numpy as jnp
from jax.lax import scan
from jax.random import PRNGKey

from abbfn2.bfn import BFN, MultimodalBFN
from abbfn2.bfn.types import ThetaMM
from abbfn2.sample.schedules import TimeScheduleFn


@dataclass
class SDESampleFn:
    """Creates the SDE sampling function for the BFN model.

    Args:
        bfn (BFN): The BFN model.
        num_steps (int): The number of steps to iterate for generating samples.
        time_schedule (TimeScheduleFn): The time schedule function.
        greedy (bool): Whether to sample the mode of the distribution (greedy) or sample from the distribution. Defaults to True.
        max_score (float | None): Range at which to clip the conditional score. Defaults to 1.0. Set to None for no clipping (can lead to unstable sampling)
        naive (bool): Reverts to naive inpainting if True (i.e. doesn't use the conditional score, only the conditioning data). Defaults to False
        mask_receiver_sample (bool): Controls whether to mask the receiver sample with the sender sample for the conditioning data. Defaults to True
    """

    bfn: BFN
    num_steps: int
    time_schedule: TimeScheduleFn
    greedy: bool = True
    max_score: float | None = 1.0
    naive: bool = False
    mask_receiver_sample: bool = True
    score_scale: float | None = None

    def __post_init__(self):
        """Check that the BFN model is multimodal."""
        assert isinstance(
            self.bfn,
            MultimodalBFN,
        ), "Sampling function only supports multimodal BFN, not\n" + str(self.bfn)
        assert (self.max_score is None) or (
            self.max_score > 0
        ), "max_score must either be None or > 0, not\n" + str(self.max_score)

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
            params (Params): The network params.
            x (Dict[str, Array]): The "ground-truth" sample on which to condition generation.
            mask_sample (Dict[str, Array]): Variable-wise mask determining which regions of the ground-truth
                is visible during sampling. i.e. 0 means the network does not know the ground-truth.


        Returns:
            Array: The generated samples.

        Note:
            This function implements conditional SDE based sampling similar to diffusion models in "Score-Based Generative Modeling through Stochastic Differential Equations", Song et. al.
            by adapting using the BFN SDE formulation from "Unifying Bayesian Flow Networks and Diffusion Models through Stochastic Differential Equations", Xue et. al.
        """
        # Prepare model input.
        theta = self.bfn.get_prior_input_distribution()
        run_net = partial(
            self.bfn.apply_output_network,
            params,
            mask=None,
        )

        def conditional_log_prob_fn(
            key: PRNGKey, theta: ThetaMM, t: float, alpha: dict[str, Array]
        ):
            key_output, key_receiver = jax.random.split(key, 2)
            pred = run_net(key_output, theta, t)

            log_prob = self.bfn.conditional_log_prob(
                pred, x, jax.tree_map(lambda arr: arr > t, mask_sample), jax.lax.stop_gradient(theta)
            )

            #log_prob = self.bfn.conditional_log_prob(
            #    pred, x, mask_sample, jax.lax.stop_gradient(theta)
            #)

            y = self.bfn.sample_receiver_distribution(pred, alpha, key_receiver)
            return log_prob, y


        conditional_score_fn = jax.grad(
            conditional_log_prob_fn, argnums=1, has_aux=True
        )

        def loop_body(theta: ThetaMM, xs: tuple[int, PRNGKey]):
            # i starts from 1 in paper, from 0 here
            i, key = xs
            key_output, key_receiver, key_sender = jax.random.split(key, 3)
            t_start, t_end = self.time_schedule(i, self.num_steps)
            beta = self.bfn.compute_beta(params, t_start)
            beta_next = self.bfn.compute_beta(params, t_end)
            alpha = jax.tree_map(lambda b2, b1: b2 - b1, beta_next, beta)

            if (x is None) or self.naive:
                pred = run_net(key_output, theta, t_start)
                y_rec = self.bfn.sample_receiver_distribution(pred, alpha, key_receiver)
                conditional_score = None
            else:
                conditional_score, y_rec = conditional_score_fn(
                    key_output, theta, t_start, alpha
                )
                if self.max_score is not None:
                    conditional_score = jax.tree_map(
                        lambda s: jnp.clip(s, -self.max_score, self.max_score),
                        conditional_score,
                    )

                if isinstance(self.score_scale, float):
                    conditional_score = jax.tree_map(
                        lambda arr: arr * self.score_scale,
                        conditional_score,
                    )

                elif self.score_scale is not None:

                    def apply_scale(score, key):
                        scale = self.score_scale[
                            key
                        ]  # Get the scale for the corresponding key
                        return jax.tree_map(
                            lambda x: x * scale, score
                        )  # Scale all arrays within the object

                    self.score_scale = dict(self.score_scale)
                    for dm in conditional_score.keys():
                        if dm not in self.score_scale:
                            self.score_scale[dm] = 1.0

                    conditional_score = {
                        dm: apply_scale(score, dm)
                        for dm, score in conditional_score.items()
                    }

            if (self.mask_receiver_sample is False) or (x is None):
                y = y_rec
            else:
                y_sen = self.bfn.sample_sender_distribution(x, alpha, key_sender)
                # Choose between samples from receiver and sender distibutions depending on the mask.
                # Note: we add ending dimensions to the mask such that is can be broadcast over y_sen, y_rec
                y = jax.tree_util.tree_map(
                    lambda y_sen, y_rec, m: jnp.where(
                        m.reshape(m.shape + (1,) * (y_sen.ndim - m.ndim)),
                        y_sen,
                        y_rec,
                    ),
                    y_sen,
                    y_rec,
                    jax.tree_map(lambda arr: arr > t_start, mask_sample),
                    #mask_sample,
                )

            theta = self.bfn.update_distribution(
                theta, y, alpha, conditional_score,
                jax.tree_map(lambda arr: arr > t_start, mask_sample),
            )

            #theta = self.bfn.update_distribution(
            #    theta, y, alpha, conditional_score, mask_sample
            #)
            return theta, y

        # Run sampling for t=0, 1/N, ..., (N-1)/N.
        key, ks = jax.random.split(key)

        theta, ys = scan(
            loop_body,
            theta,
            (jnp.arange(self.num_steps), jax.random.split(ks, self.num_steps)),
        )

        # Run the output network for t=1.
        key, key_output = jax.random.split(key)

        pred = self.bfn.apply_output_network(
            params,
            key_output,
            theta,
            t=1,
            mask=None,
        )
        # Sample from the output network.
        # Note that DiscretizedBFN predictions require the number of bins to be passed.
        ks = jax.random.split(key, len(self.bfn.data_modes))
        args = {dm: {} for dm in self.bfn.bfns}

        if self.greedy:

            # Sample the mode of the distribution.
            sample = {
                dm: pred.to_distribution(**args[dm]).mode() for dm, pred in pred.items()
            }
        else:
            # Sample from the distribution.
            sample = {
                dm: pred.to_distribution(**args[dm]).sample(seed=k)
                for (dm, pred), k in zip(pred.items(), ks)
            }

        return sample, pred
