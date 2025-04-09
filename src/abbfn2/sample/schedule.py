from dataclasses import dataclass


@dataclass
class LinearScheduleFn:
    """A linear time schedule for sampling a BFN."""

    def __call__(self, i: int, num_steps: int, eta: float = 0.0) -> tuple[float, float]:
        """Generate the sample time interval for the given iteration.

        Args:
            i (int): The iteration index.  From 0 to num_steps - 1.
            num_steps (int): The number of steps in the schedule.
            eta (float): Sample time in [η, 1]. η ≠ 0.0 for ODE/SDE Solvers.

        Returns:
            Tuple[float, float]: The start and end times for the sample interval.
        """
        t_start = eta + i * (1.0 - eta) / num_steps
        t_end = eta + (i + 1) * (1.0 - eta) / num_steps

        return t_start, t_end
