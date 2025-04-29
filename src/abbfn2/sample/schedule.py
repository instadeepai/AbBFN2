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
