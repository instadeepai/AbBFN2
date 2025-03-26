import math
from typing import Any

import jax.numpy as jnp
from jax import Array

from abbfn2.bfn.output_network.noise_schedules import NoiseSchedule


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

    def init(self, key: Array):
        """Initializes the schedule with a fixed beta(1) value.

        Is a no-op for this schedule as it does not require any parameters or state.
        """
        return {}

    def gamma(self, params: Any, t: Array) -> Array:
        """Computes the gamma (γ) values for the given timesteps.

        Functional form:
            γ(t) = β(t) / (1 + β(t))

        Args:
            params (Any): Ignored in this implementation.
            t (Array): An array of timesteps at which to compute the gamma values.

        Returns:
            Array: The gamma values for the specified timesteps.
        """
        gamma = 1 - jnp.power(self._sigma_1_sq, t)
        return gamma

    def beta(self, params: Any, t: Array) -> Array:
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

    def alpha(self, params: Any, t: Array) -> Array:
        """Computes the alpha (α) values for the given timesteps.

        Functional form:
            α(t) = - 2ln(σ1) / σ1^{2t}

        Args:
            params (Any): Ignored in this implementation.
            t (Array): An array of timesteps at which to compute the alpha values.

        Returns:
            Array: The alpha values for the specified timesteps.
        """
        alpha = -2 * math.log(self.sigma_1) / jnp.power(self._sigma_1_sq, t)
        return alpha
