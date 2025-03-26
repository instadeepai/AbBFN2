from abc import ABC, abstractmethod
from typing import Any
from jax import Array

class NoiseSchedule(ABC):
    """An abstract base class for defining noise schedules for a BFN."""

    @abstractmethod
    def init(self, key: Array) -> Any:
        """Initializes any parameters or state required by the noise schedule.

        Args:
            key (Array): A JAX random key used for initializing parameters, if necessary.

        Returns:
            An optional state or parameters object that will be passed to other methods of the schedule.
        """
        pass

    @abstractmethod
    def beta(self, params: Any | None, t: Array) -> Array:
        """Computes the beta (β) values for the given timesteps.

        Args:
            params (Optional[Any]): Parameters or state initialized by `init`, if any.
            t (Array): An array of timesteps at which to compute the beta values.

        Returns:
            Array: The beta values for the specified timesteps.
        """
        pass

    @abstractmethod
    def alpha(self, params: Any | None, t: Array) -> Array:
        """Computes the alpha (α) values for the given timesteps.

        Args:
            params (Optional[Any]): Parameters or state initialized by `init`, if any.
            t (Array): An array of timesteps at which to compute the alpha values.

        Returns:
            Array: The alpha values for the specified timesteps.
        """
        pass
