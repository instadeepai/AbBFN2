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

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from distrax import Categorical, Distribution, Normal
from flax import struct
from jax import Array


@struct.dataclass
class Theta(ABC):
    """Parameters of the distribution modelled by the BFN.

    Each modality may be parameterised differently, hence the contents
    of this dataclass are undefined.
    """

    pass

    @abstractmethod
    def to_distribution(self) -> Distribution:
        """Return the parameterised distribution.

        Note that, in general, this is the input distribution to the BFN.

        Returns:
            Distribution: A distrax distribution.
        """
        pass

    @abstractmethod
    def get_normalised_entropy(
        self,
        norm_factor: float | Array | None = None,
    ) -> Array:
        """Return a per-variable measure of the entropy of the input distribution.

        Args:
            norm_factor: Optional[Union[float, Array]]: A factor to inversly scale the
            returned entropy by.  Can be a None, a single number or a per-variable array.

        Returns:
            Array: A (possibly normalised) per variable measure of entropy.
        """
        pass


@struct.dataclass
class ThetaDiscrete(Theta):
    """Parameters of the discrete distribution modelled by the BFN.

    Args:
        logits (Array): Logits over variables with K classes (shape [...var_shape..., K])
    """

    logits: Array

    def to_distribution(self) -> Categorical:
        """Return the parameterised distribution.

        Returns:
            Categorical: The parameterised categorical distribution.
        """
        return Categorical(logits=self.logits)

    def get_normalised_entropy(
        self,
        norm_factor: float | Array | None = None,
    ) -> Array:
        """Return a per-variable measure of the entropy of the input distribution.

        Args:
            norm_factor: Optional[Union[float, Array]]: A factor to inversely scale the
            returned entropy by.  Can be a None, a single number or a per-variable array.
            If None; the output is normalised by the maximum possible entropy.

        Returns:
            Array: A (possibly normalised) per variable measure of entropy.
        """
        # Calculate entropy per input variable (i.e. shape [...var_shape...])
        entropy = self.to_distribution().entropy()

        if norm_factor is None:
            # Normalise entropy by maximum possible entropy (H_max = -log(1/K) = log(K)).
            norm_factor = jnp.log(self.logits.shape[-1])

        entropy /= norm_factor

        return entropy


@struct.dataclass
class ThetaContinuous(Theta):
    """Parameters of a continuous input distribution modelled by the BFN.

    Args:
        mu (Array): Mean of distribution for each variable and dimension (shape [...var_shape...]).
        rho (Array): Precision of distribution for each variable (shape [...var_shape...]).
    """

    mu: Array
    rho: Array

    def to_distribution(self) -> Normal:
        """Return the parameterised distribution.

        Returns:
            distrax.Categorical: The parameterised categorical distribution.
        """
        # Note sqrt as Distrax.Normal has scale = s.d. of distribution whereas N(mu, sigma) has sigma referring to variance.
        return Normal(loc=self.mu, scale=1 / self.rho**0.5)

    def get_normalised_entropy(
        self,
        norm_factor: float | Array | None = None,
    ) -> Array:
        """Return a per-variable measure of the entropy of the input distribution.

        Args:
            norm_factor: Optional[Union[float, Array]]: A factor to inversely scale the
            returned entropy by.  Can be a None, a single number or a per-variable array.
            If None; the output is normalised through a sigmoid (i.e. H_norm = sigmoid(H)).

        Returns:
            Array: A (possibly normalised) per variable measure of entropy (shape [...var_shape...]).
        """
        # Calculate per-variable entropy (shape [N]).
        # Note that entropy of normal distribution x∼N(μ,σ^2) is H(x) = 1/2 * log(2π*σ^2) + 1/2.
        sigma_sq = 1 / self.rho
        entropy = 0.5 * jnp.log(2 * math.pi * sigma_sq) + 0.5

        if norm_factor is None:
            # Normalise entropy through sigmoid to restrict range between 0 and 1.
            scaled_entropy = jax.nn.sigmoid(entropy)

        return scaled_entropy


ThetaMM = dict[str, Theta]


@struct.dataclass
class OutputNetworkPrediction:
    """The prediction of the output network.

    The prediction of the output network may be interpreted differently for each
    modality,  hence the contents of this dataclass are undefined.
    """

    pass

    @abstractmethod
    def to_distribution(self) -> Distribution:
        """Return the parameterised distribution.

        Note that, in general, this is the output distribution of the BFN.

        Returns:
            Distribution: A distrax distribution.
        """
        pass


@struct.dataclass
class OutputNetworkPredictionDiscrete(OutputNetworkPrediction):
    """Predicted parameters of the output distribution.

    Args:
        logits (Array): Logits over N variables with K classes (shape [N, K])
    """

    logits: Array

    def to_distribution(self) -> Categorical:
        """Return the parameterised distribution.

        Returns:
            Categorical: The parameterised categorical distribution.
        """
        return Categorical(logits=self.logits)


@struct.dataclass
class OutputNetworkPredictionContinuous(OutputNetworkPrediction):
    """The prediction of the output network for a continuous-variable BFN.

    Specifically, the output network predicts the ground truth data, x.

    Args:
        x (Array): Prediction of the data of shape [...var_shape...].
        rho (Array): Per-variable precision parameter (shape [...var_shape...]).

    Note:
        The precision parameter is fully defined by the noise schedule, ρ = 1 + β(t),
        and therefore not modelled by the network.  Therefore, ρ is defaulted to None,
        however it is required to be set in order for the output distribution to be
        calculated.
    """

    x: Array
    rho: Array | None = None

    def to_distribution(self) -> Normal:
        """Return the parameterised distribution.

        Returns:
            distrax.Categorical: The parameterised categorical distribution.
        """
        assert (
            self.rho is not None
        ), "Precision parameter (ρ) must be set to convert to distribution."
        return Normal(
            loc=self.x,
            scale=1 / self.rho**0.5,
        )  # Note: Normal has scale = s.d. hence sqrt.


OutputNetworkPredictionMM = dict[str, OutputNetworkPrediction]
