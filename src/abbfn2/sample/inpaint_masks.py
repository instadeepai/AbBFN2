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

"""Functions for defining masking regimes for inpainting experiments.

Note that the convention we use here is the same as in the inpainting
functions in abbfn2.sample.functions - 1 is used to indicate an element which
is visible during the sampling (generation) process, while 0 is used for elements
which are hidden (to be generated).
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from abbfn2.data_mode_handler import DataModeHandler


def get_mask(x: Array, dm_handler: DataModeHandler, gt_visible: bool = False) -> Array:
    """Gets a mask for the input array using its data mode handler.

    Gets a mask for the input array using its data mode handler via the
    sample_to_mask() function. If gt_visible is False, this returns a mask which is
    the mask from DataModeHandler inverted, since sample_to_mask() returns a mask
    using 1 to indicate valid elements of the data mode.

    If gt_visible is True, just returns a mask of all 1s (i.e. no masking).

    Args:
        x (Array): The input array.
        dm_handler (DataModeHandler): The data mode handler for the input array.
        gt_visible (bool): Whether the ground truth is visible. Defaults to False.
    """
    if gt_visible:
        return jnp.ones(x.shape, dtype=jnp.int32)
    mask = dm_handler.sample_to_mask(x)
    return 1 - mask


@dataclass
class InpaintMaskGenerationFn:
    """A base class for functions generating inpainting masks.

    This class defines a callable interface for generating masks based on given samples and data mode handlers.
    Derived classes must implement the __call__ method to specify the mask generation logic.
    """

    def __call__(
        self,
        samples: dict[str, Array],
        dm_handlers: dict[str, DataModeHandler],
    ) -> Array:
        """Generates a mask for inpainting.

        Args:
            key: A pseudo-random number generator key used for random operations.
            samples: A dictionary of samples where keys are data modes, and values are the corresponding samples.
            dm_handlers: A dictionary mapping data modes to their respective DataModeHandler objects.

        Returns:
            An Array representing the generated mask.  When passed to InpaintFn, the mask determines which elements
            of the ground truth data are visible (unmasked <-> 1) and which are to be predicted (masked <-> 0).
        """
        pass


@dataclass
class PredictDataModeMaskFn(InpaintMaskGenerationFn):
    """Generates a mask for inpainting to predict the specified data modes.

    Attributes:
        data_modes: A string or list of strings specifying the data modes to be masked for prediction.
    """

    data_modes: str | list[str]

    def __post_init__(self):
        """Ensures data_modes is a list for consistent processing."""
        self.data_modes = list(self.data_modes)

    def __call__(
        self,
        samples: dict[str, Array],
        dm_handlers: dict[str, DataModeHandler] = None,
    ) -> dict[str, Array]:
        """Generates a prediction mask for specified data modes.

        Args:
            key: A pseudo-random number generator key used for random operations.
            samples: A dictionary of samples where keys are data modes, and values are the corresponding samples.
            dm_handlers: An optional dictionary mapping data modes to their respective DataModeHandler objects.

        Returns:
            A dictionary of masks, where keys are data modes, and values are the corresponding masks.
        """
        logging.info(f"Generating prediction masks for data modes {self.data_modes}.")

        def get_mask_for_sample(sample: dict[str, Array]) -> dict[str, Array]:
            mask = {}
            for dm, x in sample.items():
                # if the datamode is not in the list of predict datamodes, return a mask of all 1s
                mask[dm] = get_mask(
                    x,
                    dm_handlers[dm],
                    gt_visible=dm not in self.data_modes,
                )
            return mask

        masks = jax.vmap(get_mask_for_sample)(samples)

        return masks


@dataclass
class ConditionDataModeMaskFn(InpaintMaskGenerationFn):
    """Generates a mask for inpainting to condition the generation on the specified data modes.

    Attributes:
        data_modes: A string or list of strings specifying the data modes to be conditioned upon. Data modes not
                    listed are assumed to be the target for prediction.
    """

    data_modes: str | list[str]

    def __post_init__(self):
        """Ensures data_modes is a list for consistent processing across different uses."""
        self.data_modes = list(self.data_modes)

    def __call__(
        self,
        samples: dict[str, Array],
        dm_handlers: dict[str, DataModeHandler] = None,
    ) -> dict[str, Array]:
        """Generates a mask for the samples dict indicating which data modes are conditioned upon.

        Args:
            key: A pseudo-random number generator key used for random operations. Note: In this context, the key
                 may not be directly used since the masking logic is deterministic based on the specified data modes.
            samples: A dictionary of samples where keys are data modes, and values are the corresponding samples.
            dm_handlers: An optional dictionary mapping data modes to their respective DataModeHandler objects. This
                         parameter may not be directly used but is included for interface consistency.

        Returns:
            A dictionary of masks, where keys are data modes, and values are the corresponding masks.
        """
        logging.info(f"Generating conditioning masks for data modes {self.data_modes}.")

        def get_mask_for_sample(sample: dict[str, Array]) -> dict[str, Array]:
            mask = {}
            for dm, x in sample.items():
                # if the datamode is in the list of conditional datamodes, return a mask of all 1s
                mask[dm] = get_mask(
                    x,
                    dm_handlers[dm],
                    gt_visible=dm in self.data_modes,
                )
            return mask

        masks = jax.vmap(get_mask_for_sample)(samples)

        return masks
