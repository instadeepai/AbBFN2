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

from abbfn2.data_mode_handler.base import DataModeHandler
from abbfn2.data_mode_handler.oas_paired.constants import (
    VALID_REGION_LENGTHS as VALID_CDR_LENGTHS,
)
from abbfn2.data_mode_handler.utils import load_from_hdf5, write_to_hdf5


class CDRLengthsDataModeHandler(DataModeHandler):
    """Class for CDR length data mode handler."""

    def __init__(self, dm_key: str, unknown_label: int = -1):
        """Initialise the DM handler with dicts to convert lengths to classes.

        Args:
            dm_key (str): The key by which to identify the CDR loop to be processed.
            unknown_label (int): The ID to be used for missing CDR lengths.
                Defaults to -1.
        """
        self.dm_key = dm_key
        self.lens_list = list(VALID_CDR_LENGTHS[dm_key].keys())
        self.len2id = {length: idx for idx, length in enumerate(self.lens_list)}
        self.id2len = {idx: k for k, idx in self.len2id.items()}

        # Define an "unknown" label for lengths not in the valid list.
        self.unknown_id = len(self.len2id)
        self.unknown_label = unknown_label
        self.len2id[self.unknown_label] = self.unknown_id
        self.id2len[self.unknown_id] = self.unknown_label

    def sample_to_data(self, sample: Array) -> Array:
        """Converts an array of sample indices to a list of length labels.

        This method takes an array of indices and maps each index to its corresponding
        CDR length label as defined in the id2len dictionary attribute of the class.
        It supports handling of both scalar values and multidimensional arrays,
        automatically squeezing them to a 1D list if necessary. If the input sample
        is not an array or has more than one dimension after squeezing, a ValueError
        is raised.

        Args:
            sample (Array): A numpy array of sample indices. Can be a scalar, 1D, or
                multi-dimensional array.

        Raises:
            ValueError: If the input sample has more than one dimension after attempting
                to squeeze it.

        Returns:
            List[int]: A list of CDR length labels corresponding to the input sample
                indices.
        """
        inp_shp = sample.shape
        if sample.ndim > 1:
            sample = sample.squeeze()

        if sample.ndim == 0:
            sample = sample[None]
        elif sample.ndim > 1:
            raise ValueError(f"Sample has unexpected shape: {inp_shp}")

        # Convert to length labels.
        return np.array(
            [self.id2len.get(int(idx), self.unknown_label) for idx in sample],
            dtype="int",
        )

    def data_to_sample(self, data: Array) -> Array:
        """Converts a list of CDR length labels to a numpy array of sample indices.

        Args:
            data: A list of CDR length labels.

        Returns:
            Array: A numpy array of sample indices.
        """
        return np.array([[self.len2id.get(lab, self.unknown_id)] for lab in data])

    def save_data(
        self,
        data: np.ndarray,
        dir_path: Path,
        name: str = "properties.hdf5",
        exists_ok: bool = True,
    ) -> None:
        """Save data into a hdf5 file."""
        path = dir_path / name
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")

        logging.info(f"Saving {self.dm_key} labels to {path}.")
        data = {self.dm_key: data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> Array:
        """Load data from a hdf5 file."""
        path = dir_path / name
        data = load_from_hdf5(path, keys=[self.dm_key])
        return data[self.dm_key]
