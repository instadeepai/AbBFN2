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
from abbfn2.data_mode_handler.utils import load_from_hdf5, write_to_hdf5


class LLocusDataModeHandler(DataModeHandler):
    """Handles data mode specific to labelling the locus of the light chain."""

    dm_key = "l_locus"

    def __init__(self, dm_key: str = "l_locus", unknown_label: str = "unknown"):
        """Initialise the DM handler for light chain locus.

        Args:
            dm_key (str): The key by which to identify the light chain locus.
            unknown_label (str): The label to be used for missing loci.
                Defaults to "unknown".
        """
        self.L_LOCI = ["K", "L"]
        self.dm_key = dm_key

        self.locus2id = {k: idx for idx, k in enumerate(self.L_LOCI)}
        self.id2locus = {idx: k for k, idx in self.locus2id.items()}

        self.unknown_id = len(self.L_LOCI)
        self.unknown_label = unknown_label
        self.locus2id[self.unknown_label] = self.unknown_id
        self.id2locus[self.unknown_id] = self.unknown_label

    def sample_to_data(self, sample: Array) -> list[str]:
        """Converts an array of sample indices to a list of locus labels.

        This method takes an array of indices and maps each index to its corresponding
        locus label as defined in the id2locus dictionary attribute of the class.
        It supports handling of both scalar values and multidimensional arrays,
        automatically squeezing them to a 1D list if necessary. If the input sample
        is not an array or has more than one dimension after squeezing, a ValueError
        is raised.

        Args:
            sample (Array): A numpy array of sample indices. Can be a scalar, 1D, or multi-dimensional array.

        Returns:
            List[str]: A list of locus labels corresponding to the input sample indices.

        Raises:
            ValueError: If the input sample has more than one dimension after attempting to squeeze it.
        """
        # Ensure sample is 1D list of locus labels.
        inp_shp = sample.shape
        if sample.ndim > 1:
            sample = sample.squeeze()

        if sample.ndim == 0:
            sample = sample[None]
        elif sample.ndim > 1:
            raise ValueError(f"Sample has unexpected shape: {inp_shp}")

        # Convert to locus labels.
        return [self.id2locus[int(idx)] for idx in sample]

    def data_to_sample(self, data: list[str]) -> Array:
        """Converts a list of locus labels to a numpy array of sample indices.

        Args:
            data: A list of locus labels.

        Returns:
            Array: A numpy array of sample indices.
        """
        return np.array([[self.locus2id.get(lab, self.unknown_id)] for lab in data])

    def save_data(
        self,
        data: list[str],
        dir_path: Path,
        name: str = "properties.hdf5",
        exists_ok: bool = True,
    ) -> None:
        """Saves a list of locus labels to a file, each on a new line.

        Args:
            data (List[str]): The list of locus labels to save.
            dir_path (Path): The directory to save the data to.
            name (str): The name of the HDF5 file. Defaults to "properties.hdf5".
            exists_ok (bool): If False, raise an error if the file already exists. Defaults to True.

        Raises:
            FileExistsError: If the file already exists and exists_ok is False.
            IOError: If there is an issue writing to the file.
        """
        path = dir_path / name

        # Check if the file exists and handle according to exists_ok flag
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")

        logging.info(f"Saving light chain locus labels to {path}.")
        data = {self.dm_key: data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)
        # with path.open("w") as file:
        #     for item in data:
        #         file.write(f"{item}\n")

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> Array:
        """Loads data from an HDF5 file.

        Args:
            dir_path (Path): The directory in which the data is stored.
            name (str): The name of the HDF5 file. Defaults to "properties.hdf5".

        Returns:
            Dict: The loaded data.
        """
        path = dir_path / name
        data = load_from_hdf5(path, keys=[self.dm_key])
        return data[self.dm_key]
