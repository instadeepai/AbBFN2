from abc import ABC, abstractmethod

from flax import linen as nn
from jax import Array

from abbfn2.bfn.types import Theta


class DistributionEncoding(ABC, nn.Module):
    """An abstract base class for encoding distribution information into embeddings."""

    @abstractmethod
    def __call__(self, x: Array, theta: Theta) -> Array:
        """Applies time encoding to the input embeddings.

        Note that the expectation is that the output will have the same shape as the input x.

        Args:
            x (Array): The input embeddings.
            theta (Theta): The input distribution.

        Returns:
            Array: The embeddings with time encoding applied.
        """
        pass


class NoOpEncoding(DistributionEncoding):
    """A no-operation (NoOp) time encoding that returns the input without any modification."""

    def __call__(self, x: Array, theta: Theta) -> Array:
        """Returns the input embeddings unchanged, ignoring the distribution input.

        Args:
            x (Array): The input embeddings.
            theta (Theta): The input distribution.

        Returns:
            Array: The same input array `x`, unmodified.
        """
        return x


class EntropyEncoding(DistributionEncoding):
    """An encoding that embeds the per-variable entropy of the input distribution into the input embeddings.

    Args:
        with_bias (bool): Whether to include a bias term in the projection.
        zero_init (bool): Whether to initialize the projection weights to zero.
        name (str): The module name.
    """

    with_bias: bool = False
    zero_init: bool = False
    name: str = "entropy_encoding"

    @nn.compact
    def __call__(self, x: Array, theta: Theta) -> Array:
        """Embeds the per-variable entropy of the input distribution into the input embeddings.

        Args:
            x (Array): The input embeddings.
            theta (Theta): The input distribution.

        Returns:
            Array: The result of adding the projected input embeddings and entropy embeddings.
        """
        entropy = theta.get_normalised_entropy()

        # entropy: [...var_shape...] -> entropy_embedding: [...var_shape..., inp_dim]
        entropy_embedding = nn.Dense(
            x.shape[-1],
            use_bias=self.with_bias,
            kernel_init=nn.initializers.constant(0.0)
            if self.zero_init
            else nn.initializers.lecun_normal(),
        )(entropy[..., None])

        return x + entropy_embedding
