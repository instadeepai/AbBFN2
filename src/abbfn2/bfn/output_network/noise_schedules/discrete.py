from typing import Any

from jax import Array

from abbfn2.bfn.output_network.noise_schedules import NoiseSchedule


class FixedDiscreteSchedule(NoiseSchedule):
    """A noise schedule for discrete data with fixed beta(1)."""

    def __init__(self, beta_1: float = 1.0):
        """Initializes the schedule with a fixed beta(1) value.

        Args:
            beta_1 (float, optional): The fixed beta(1) value. Defaults to 1.0.
        """
        self.beta_1 = beta_1

    def init(self, key: Array):
        """Initializes the schedule with a fixed beta(1) value.

        Is a no-op for this schedule as it does not require any parameters or state.
        """
        return {}

    def beta(self, params: Any, t: Array) -> Array:
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

    def alpha(self, params: Any, t: Array) -> Array:
        """Computes the alpha (α) values for the given timesteps.

        Functional form:
            α(t) = 2 * β1 * t

        Args:
            params (Any): Ignored in this implementation.
            t (Array): An array of timesteps at which to compute the alpha values.

        Returns:
            Array: The alpha values for the specified timesteps.
        """
        return 2 * t * self.beta_1
