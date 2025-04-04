from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jax import Array
from jax.random import PRNGKey

from abbfn2.bfn import BFN, MultimodalBFN
from abbfn2.bfn.types import OutputNetworkPredictionMM
from abbfn2.sample.schedules import TimeScheduleFn


@dataclass
class BaseSampleFn(ABC):
    """Get the sampling function for the BFN model.

    Args:
        bfn (BFN): The BFN model.
        num_steps (int): The number of steps to iterate for generating samples.
        time_schedule (TimeScheduleFn): The time schedule function.
        greedy (bool): Whether to sample the mode of the distribution (greedy) or sample from the distribution. Defaults to True.
        use_self_conditioning (bool): Whether to condition on the previous prediction when generating the next prediction. Defaults to False.
    """

    bfn: BFN
    num_steps: int
    time_schedule: TimeScheduleFn
    greedy: bool = True
    use_self_conditioning: bool = False

    def __post_init__(self):
        """Check that the BFN model is multimodal."""
        assert isinstance(
            self.bfn,
            MultimodalBFN,
        ), "Sampling function only supports multimodal BFN."

    def _sample_from_network_prediction(
        self, key: PRNGKey, pred: OutputNetworkPredictionMM
    ) -> dict[str, Array]:
        """Sample from the network prediction.

        Args:
            key (PRNGKey): The random key used for generating samples.
            pred (OutputNetworkPredictionMM): The network prediction.

        Returns:
            Dict[str, Array]: The sampled values.
        """
        args = {dm: {} for dm in self.bfn.bfns}


        if self.greedy:
            # Sample the mode of the distribution.
            sample = {
                dm: pred.to_distribution(**args[dm]).mode() for dm, pred in pred.items()
            }
        else:
            # Sample from the distribution.
            ks = jax.random.split(key, len(self.bfn.data_modes))
            sample = {
                dm: pred.to_distribution(**args[dm]).sample(seed=k)
                for (dm, pred), k in zip(pred.items(), ks)
            }

        return sample

    @abstractmethod
    def __call__(
        self,
        key: PRNGKey,
        params: dict,
        x: dict[str, Array] | None,
        mask_sample: dict[str, Array] | None,
    ) -> dict[str, Array]:
        """Generate samples by "hallucinating" Alice.

        Parameters:
            key (PRNGKey): The random key used for generating samples.
            params (dict): The network params.
            x (Dict[str, Array] | None): The "ground-truth" sample on which to condition generation.
            mask_sample (Dict[str, Array] | None): Variable-wise mask determining which regions of the ground-truth
                is visible during sampling. i.e. 0 means the network does not know the ground-truth.

        Returns:
            dict[str, Array]: The generated samples.
        """
