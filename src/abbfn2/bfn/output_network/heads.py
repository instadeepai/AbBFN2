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

import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from abbfn2.bfn.types import (
    OutputNetworkPrediction,
    OutputNetworkPredictionContinuous,
    OutputNetworkPredictionDiscrete,
    Theta,
    ThetaContinuous,
    ThetaDiscrete,
)


class EntropyEncoding(nn.Module):
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
            kernel_init=(
                nn.initializers.constant(0.0)
                if self.zero_init
                else nn.initializers.lecun_normal()
            ),
        )(entropy[..., None])

        return x + entropy_embedding


class Encoder(ABC, nn.Module):
    """A module for encoding input data.

    This encoder applies a linear transformation followed by optional time and positional encodings.
    """

    @abstractmethod
    def __call__(
        self,
        theta: Theta,
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.

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
        distribution_encoding (EntropyEncoding): The distribution encoding to use. Defaults to None.
        with_bias (bool): Whether to include a bias term in the linear transformation. Defaults to True.
        name (str): The name of the module. Defaults to "encoder_discrete".
    """

    output_dim: int
    normalise_input: bool = True
    distribution_encoding: EntropyEncoding | None = None
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
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.

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
        distribution_encoding (EntropyEncoding): The distribution encoding to use. Defaults to None.
        with_bias (bool): Whether to include a bias term in the linear transformation. Defaults to True.
        name (str): The name of the module. Defaults to "encoder_cts".
    """

    output_dim: int
    distribution_encoding: EntropyEncoding | None = None
    with_bias: bool = True
    name = "encoder_cts"
    kwargs: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(
        self,
        theta: ThetaContinuous,
    ) -> tuple[Array, Any]:
        """Encodes input data into a dense representation.

        Args:
            theta (Theta): Parameters of the input distribution.

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


class RegressionHead(nn.Module):
    """MLP Block used in the Decoders."""

    hidden_dim: int
    output_dim: int
    with_bias: bool = True
    name: str | None = None

    """
    Args:
        hidden_dim (int): The number of hidden units.
        output_dim (int): The number of output units.
        with_bias (bool): Whether to use bias in the linear layers. Defaults to True.
        name (Optional[str]): The module name.
    """

    @nn.compact
    def __call__(self, x):
        """MLP Block used in the Decoders."""
        x = nn.LayerNorm(
            feature_axes=-1,
            epsilon=1e-5,
            use_scale=True,
            use_bias=self.with_bias,
            use_fast_variance=False,
        )(x)
        x = nn.Dense(self.hidden_dim, use_bias=self.with_bias)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.LayerNorm(
            feature_axes=-1,
            epsilon=1e-5,
            use_scale=True,
            use_bias=self.with_bias,
            use_fast_variance=False,
        )(x)
        x = nn.Dense(self.output_dim, use_bias=self.with_bias)(x)
        return x


class Decoder(ABC, nn.Module):
    """A module for decoding from the output network."""

    @abstractmethod
    def __call__(
        self,
        x: Array,
        skip_args: Any | None,
        t: float | None,
        beta: Array | None,
    ) -> OutputNetworkPrediction:
        """Encodes input data into a dense representation.

        Args:
            x (Array): The embeddings to decode (shape [...var_shape..., N_dim]).
            skip_args (Optional[Any]): Additional state or parameters that may be required for decoding.
            t (Optional[float]): The time at which the input distribution was computed.
            beta (Optional[Array]): A per-variable value of the accuracy schedule (shape [...var_shape...]).

        Returns:
            pred (OutputNetworkPrediction): The output network prediction.
        """

    pass


class DiscreteDecoder(Decoder):
    """A module for decoding data for a discrete data mode."""

    output_dim: int
    max_logits_magnitude: float = None
    name = "decoder_discrete"
    kwargs: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(
        self,
        x: Array,
        skip_args: dict[str, Array],
        t: float | None,
        beta: Array | None,
    ) -> OutputNetworkPredictionDiscrete:
        """Encodes input data into a dense representation.

        Args:
            x (Array): The embeddings to decode (shape [...var_shape..., dim])
            skip_args (dict[str, Array]): Additional state or parameters that may be required for decoding. Unused in this module.
            t (Optional[float]): The time at which the input distribution was computed.  Unused in this module.
            beta (Optional[Array]): A per-variable value of the accuracy schedule (shape [D]). Unused in this module.

        Returns:
            pred (OutputNetworkPredictionDiscrete): The output network prediction.
        """
        dim = x.shape[-1]

        logits = RegressionHead(
            hidden_dim=2 * dim,
            output_dim=self.output_dim,
        )(x)

        if self.max_logits_magnitude is not None:
            log_mag = self.max_logits_magnitude
            logits = jnp.clip(logits, a_min=-log_mag, a_max=log_mag)

        return OutputNetworkPredictionDiscrete(logits=logits)


class ContinuousDecoder(Decoder):
    """A module for decoding the output network for a continuous data mode.

    This encoder applies a linear transformation followed by optional time and positional encodings.

    Attributes:
        output_dim (int): The dimensionality of the output.
        out_lims (Optional[Tuple[float, float]]): The limits of the output. Defaults to None.
        t_min (float): The minimum time at which the output is non-zero. Defaults to 1e-6.
        name (str): The name of the module. Defaults to "decoder_cts".
    """

    out_lims: tuple[float, float] | None = None
    t_min: float = 1e-6
    name = "decoder_cts"
    kwargs: dict = dataclasses.field(default_factory=dict)

    def setup(self):
        """Setup out_lims."""
        if self.out_lims is None:
            self.processed_out_lims = None
        elif isinstance(self.out_lims, (int | float)):
            self.processed_out_lims = (-self.out_lims, self.out_lims)
        else:
            assert len(self.out_lims) == 2, "out_lims must be a scalar or a 2-tuple"
            self.processed_out_lims = self.out_lims

    @nn.compact
    def __call__(
        self,
        x: Array,
        skip_args: dict[str, Array],
        t: float | None,
        beta: Array | None,
    ) -> OutputNetworkPredictionContinuous:
        """Encodes input data into a dense representation.

        Args:
            x (Array): The embeddings to decode (shape [...var_shape..., dim]).
            skip_args (dict[str, Array]): Skipped arguments from the encoder.
            t (Optional[float]): The time at which the input distribution was computed.
            beta (Optional[Array]): A per-variable value of the accuracy schedule (shape [...var_shape...]).

        Returns:
            pred (OutputNetworkPredictionContinuous): The output network prediction.
        """
        dim = x.shape[-1]

        eps = RegressionHead(
            hidden_dim=2 * dim,
            output_dim=1,
        )(
            x
        ).squeeze(-1)

        gamma = beta / (1 + beta)
        gamma = gamma.clip(min=1e-9)  # Numerical stability in the scale comp.
        scale = jnp.sqrt((1 - gamma) / gamma)

        # x = mu / gamma - scale * eps;  for eps ~ N(0, 1)
        #   mu: [...var_shape...], gamma: [...var_shape..., 1], eps: [...var_shape..., self.output_dim]
        mu = skip_args["mu"]
        x = mu / gamma - scale * eps

        # Clip output and set to zero if t < t_min. (Note don't use jax.lax.select as it
        # doesn't correctly handle nan's in the gradients of gamma & scale.)
        x = jax.lax.cond(t < self.t_min, lambda: jnp.zeros_like(x), lambda: x)

        if self.processed_out_lims is not None:
            x = x.clip(min=self.processed_out_lims[0], max=self.processed_out_lims[1])

        return OutputNetworkPredictionContinuous(x=x, rho=1 + beta)
