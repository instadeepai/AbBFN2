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

import logging
from pathlib import Path

import numpy as np
from jax import Array
from numpy import ndarray

from abbfn2.data_mode_handler.base import DataModeHandler
from abbfn2.data_mode_handler.utils import load_from_hdf5, write_to_hdf5


def scale_values(
    values: ndarray,
    src_bounds: tuple[int | float, int | float],
    trg_bounds: tuple[int | float, int | float] = (-1.0, 1.0),
) -> ndarray:
    """Rescales values to a desired range.

    Linearly maps the input (from an input range) to the target range.
    Original values outside of the input range are clipped
    to the bounds.

    Args:
        values (ndarray): Array of values to be transformed.
        src_bounds (tuple[int | float, int| float]): Tuple of lower and upper bounds of
            the INPUT range.
        trg_bounds (tuple[int | float, int| float]): Tuple of lower and upper bounds of
            the OUTPUT range. Defaults to (-1,1).

    Returns:
        ndarray: Array of rescaled values.
    """
    min_src, max_src = src_bounds
    min_trg, max_trg = trg_bounds

    assert (
        min_src < max_src
    ), f"Provided minimum ({min_src}) is larger than provided maximum ({max_src})."
    assert (
        min_trg < max_trg
    ), f"Target minimum ({min_trg}) is larger than target maximum ({max_trg})."

    return (((values - min_src) / (max_src - min_src)) * (max_trg - min_trg)) + min_trg


class ScalarDataModeHandler(DataModeHandler):
    """Data mode handler for scalar continuous values.

    This DM handler is to be used for scalar continuous variables. It allows for scaling
    and clipping of values before they are passed to the model.
    """

    def __init__(
        self,
        dm_key: str,
        data_bounds: list[int | float],
        target_bounds: list[int | float] | None = None,
        unknown_value: int | float = 100.0,
    ):
        """Initialises the DM handler.

        If target bounds are specified, data will be scaled to lie in this range. Any
        missing values are replaced with the provided unknown_value.
        """
        self.dm_key = dm_key
        self.data_bounds = tuple(data_bounds)
        self.target_bounds = (
            tuple(target_bounds) if isinstance(target_bounds, list) else target_bounds
        )
        self.unknown_value = unknown_value

    def sample_to_data(self, sample: Array) -> Array:
        """Converts a sample from the model to data by mapping it back to the original
        interval.

        Args:
            sample (Array): Batch of samples from the model. (Shape [N, 1, 1]).

        Returns:
            Array: The array of recovered scalar values.
        """
        if self.target_bounds:
            data = scale_values(
                sample,
                src_bounds=self.target_bounds,
                trg_bounds=self.data_bounds,
            )
        else:
            pass

        if data.ndim > 2:
            raise ValueError(f"Unexpected shape in data mode: {self.dm_key}")

        return np.array(data)

    def data_to_sample(self, data: Array) -> Array:
        """Converts data to a sample for the model by mapping it onto the target
        interval.

        Args:
            data (Array): An array of scalar values.

        Returns:
            Array: Array of samples for the model.
        """
        data_min, data_max = self.data_bounds
        data = np.clip(data, data_min, data_max)

        if self.target_bounds:
            sample = scale_values(
                data,
                src_bounds=self.data_bounds,
                trg_bounds=self.target_bounds,
            )
        else:
            pass

        return sample

    def save_data(
        self,
        data: Array,
        dir_path: Path,
        name: str = "properties.hdf5",
        exists_ok: bool = True,
    ) -> None:
        """Saves the given data to an HDF5 file in the specified directory.

        Args:
            data (Dict): The data to save.
            dir_path (Path): The directory in which to save the file.
            name (str, optional): The name of the HDF5 file. Defaults to
                "properties.hdf5".
            exists_ok (bool, optional): If False, raises FileExistsError if the file
                already exists. Defaults to True.

        Raises:
            FileExistsError: If the file already exists and exists_ok is False.
        """
        path = dir_path / name

        # Check if the file exists and handle according to exists_ok flag
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")

        logging.info(f"Saving {self.dm_key} records to {path}.")
        data = {self.dm_key: data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> Array:
        """Loads data from an HDF5 file.

        Args:
            dir_path (Path): The directory in which the data is stored.
            name (str, optional): The name of the HDF5 file. Defaults to
                "properties.hdf5".

        Returns:
            Dict: The loaded data.
        """
        path = dir_path / name
        data = load_from_hdf5(path, keys=[self.dm_key])
        return data[self.dm_key]
