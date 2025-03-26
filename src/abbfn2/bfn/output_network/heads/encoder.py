import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import Any

from flax import linen as nn
from jax import Array

from abbfn2.bfn.output_network.conditioning.distribution_encoding import (
    DistributionEncoding,
)
from abbfn2.bfn.types import (
    OutputNetworkPrediction,
    OutputNetworkPredictionContinuous,
    OutputNetworkPredictionDiscrete,
    Theta,
    ThetaContinuous,
    ThetaDiscrete,
)


class Encoder(ABC, nn.Module):
    """A module for encoding input data.

    This encoder applies a linear transformation followed by optional time and positional encodings.
    """

    @abstractmethod
    def __call__(
        self,
        theta: Theta,
        t: float,
        mask: Array | None,
        pred_cond: OutputNetworkPrediction | None = None,
        t_cond: float | None = None,
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.
            t (float): The time at which the input distribution was computed, used for time encoding.
            mask (Optional[Array]): Optional mask to apply to the input data.
            pred_cond (Optional[OutputNetworkPrediction]): A prediction of the output network for self-conditioning. Defaults to None.
            t_cond (Optional[float]): The time at which `pred_cond` was computed. Defaults to None.

        Returns:
            Array: The encoded data after applying linear transformation and optional encodings, with shape [..., embed_dim].
            Any: Additional state or parameters that may be required for decoding.
        """
        pass


class DiscreteEncoder(Encoder):
    """A module for encoding input data of a discrete data mode.

    This encoder applies a linear transformation followed by optional time and positional encodings.

    Attributes:
        cfg: The configuration for the encoder.
        global_cfg: The global configuration, containing settings affecting the entire model.
        name: The name of the module.

    Args:
        output_dim (int): The dimensionality of the output.
        normalise_input (bool): Whether to normalise the input logits. Defaults to True.
        distribution_encoding (DistributionEncoding): The distribution encoding to use. Defaults to None.
        with_bias (bool): Whether to include a bias term in the linear transformation. Defaults to True.
        name (str): The name of the module. Defaults to "encoder_discrete".
    """

    output_dim: int
    normalise_input: bool = True
    distribution_encoding: DistributionEncoding | None = None
    with_bias: bool = True
    name = "encoder_discrete"
    kwargs: dict = dataclasses.field(default_factory=dict)

    def setup(self):
        """Setup Discrete Encoder."""
        if not self.normalise_input:
            warnings.warn(
                "The 'normalise_input' argument is deprecated - logits are now always normalised by default.",
                DeprecationWarning,
                stacklevel=2,
            )

    @nn.compact
    def __call__(
        self,
        theta: ThetaDiscrete,
        t: float,
        mask: Array | None,
        pred_cond: OutputNetworkPredictionDiscrete | None = None,
        t_cond: float | None = None,
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.
            t (float): The time at which the input distribution was computed, used for time encoding.
            mask (Optional[Array]): Optional mask to apply to the input data. No-op in this module.
            pred_cond (Optional[OutputNetworkPredictionDiscrete]): A prediction of the output network for self-conditioning.
            t_cond (Optional[float]): The time at which `pred_cond` was computed.

        Returns:
            Array: The encoded data after applying linear transformation and optional encodings, with shape [..., embed_dim].
            Any: Additional state or parameters that may be required for decoding.
        """
        input_distribution = theta.to_distribution()

        x = nn.Dense(self.output_dim, use_bias=self.with_bias)(input_distribution.probs)

        if self.distribution_encoding is not None:
            x = self.distribution_encoding(x, theta)

        return x, {
            "logits": input_distribution.logits,
            "x_skip": x,
        }


class ContinuousEncoder(Encoder):
    """A module for encoding input data of a continuous data mode.

    Args:
        output_dim (int): The dimensionality of the output.
        distribution_encoding (DistributionEncoding): The distribution encoding to use. Defaults to None.
        with_bias (bool): Whether to include a bias term in the linear transformation. Defaults to True.
        name (str): The name of the module. Defaults to "encoder_cts".
    """

    output_dim: int
    distribution_encoding: DistributionEncoding | None = None
    with_bias: bool = True
    name = "encoder_cts"
    kwargs: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(
        self,
        theta: ThetaContinuous,
        t: float,
        mask: Array | None,
        pred_cond: OutputNetworkPredictionContinuous | None = None,
        t_cond: float | None = None,
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.
            t (float): The time at which the input distribution was computed, used for time encoding.
            mask (Optional[Array]): Optional mask to apply to the input data. No-op in this module.
            pred_cond (Optional[OutputNetworkPredictionContinuous]): A prediction of the output network for self-conditioning.
            t_cond (Optional[float]): The time at which `pred_cond` was computed.

        Returns:
            Array: The encoded data after applying linear transformation and optional encodings, with shape [batch_size, embed_dim].
            Any: Additional state or parameters that may be required for decoding.
        """
        mu = theta.mu

        # Note mu is a single number per variable, so we need to add a dimension to it.
        #  [...var_shape...] -> [...var_shape..., 1] -> [...var_shape..., output_dim]
        x = nn.Dense(self.output_dim, use_bias=self.with_bias)(mu[..., None])

        if self.distribution_encoding is not None:
            x = self.distribution_encoding(x, theta)

        return x, {"mu": mu, "x_skip": x}