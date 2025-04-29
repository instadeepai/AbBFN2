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
from typing import Any

import jax.numpy as jnp
from jax import Array


class NoiseSchedule(ABC):
    """An abstract base class for defining noise schedules for a BFN."""

    @abstractmethod
    def init(self) -> Any:
        """Initializes any parameters or state required by the noise schedule.

        Args:
            key (Array): A JAX random key used for initializing parameters, if necessary.

        Returns:
            An optional state or parameters object that will be passed to other methods of the schedule.
        """
        pass

    @abstractmethod
    def beta(self, t: Array) -> Array:
        """Computes the beta (β) values for the given timesteps.

        Args:
            params (Optional[Any]): Parameters or state initialized by `init`, if any.
            t (Array): An array of timesteps at which to compute the beta values.

        Returns:
            Array: The beta values for the specified timesteps.
        """
        pass


class FixedContinuousSchedule(NoiseSchedule):
    """A noise schedule for continuous data with fixed beta(1)."""

    def __init__(self, beta_1: float | None = None, sigma_1: float | None = None):
        """Initializes the schedule.

        This schedule can be defined with either a fixed β1 value or a fixed σ1 value.
        Recall that σ1 to be the standard deviation of the input distribution at t = 1.  These
        values are related as σ1 = 1 / sqrt(1 + β1).  If both are provided, these values must be
        consistent.

        Args:
            beta_1 (float, optional): The fixed β1 value. Defaults to None.
            sigma_1 (float, optional): The fixed σ1 value. Defaults to None.

        Raises:
            ValueError: If neither beta_1 nor sigma_1 is specified.
            ValueError: If both beta_1 and sigma_1 are specified but are inconsistent.
        """
        if (beta_1 is None) and (sigma_1 is None):
            raise ValueError("Either beta_1 or sigma_1 must be specified.")

        elif (beta_1 is not None) and (sigma_1 is None):
            self.beta_1 = float(beta_1)
            self.sigma_1 = 1 / math.sqrt(1 + self.beta_1)

        elif (beta_1 is None) and (sigma_1 is not None):
            self.sigma_1 = float(sigma_1)
            self.beta_1 = (1 / self.sigma_1**2) - 1

        else:
            self.beta_1 = float(beta_1)
            self.sigma_1 = float(sigma_1)
            if not math.isclose(1 / math.sqrt(1 + self.beta_1), self.sigma_1):
                raise ValueError("Provided beta_1 and sigma_1 values are inconsistent.")

        self._sigma_1_sq = 1 / (1 + self.beta_1)

    def init(self):
        """Initializes the schedule with a fixed beta(1) value.

        Is a no-op for this schedule as it does not require any parameters or state.
        """
        return {}

    def beta(self, t: Array) -> Array:
        """Computes the beta (β) values for the given timesteps.

        Functional form:
            β(t) = σ1^{-2t} - 1

        Args:
            params (Any): Ignored in this implementation.
            t (Array): An array of timesteps at which to compute the beta values.

        Returns:
            Array: The beta values for the specified timesteps.
        """
        beta = (1 / jnp.power(self._sigma_1_sq, t)) - 1
        return beta


class FixedDiscreteSchedule(NoiseSchedule):
    """A noise schedule for discrete data with fixed beta(1)."""

    def __init__(self, beta_1: float = 1.0):
        """Initializes the schedule with a fixed beta(1) value.

        Args:
            beta_1 (float, optional): The fixed beta(1) value. Defaults to 1.0.
        """
        self.beta_1 = beta_1

    def init(self):
        """Initializes the schedule with a fixed beta(1) value.

        Is a no-op for this schedule as it does not require any parameters or state.
        """
        return {}

    def beta(self, t: Array) -> Array:
        """Computes the beta (β) values for the given timesteps.

        Functional form:
            β(t) = β1 * t^2

        Args:
            params (Any): Ignored in this implementation.
            t (Array): An array of timesteps at which to compute the beta values.

        Returns:
            Array: The beta values for the specified timesteps.
        """
        return t**2 * self.beta_1
