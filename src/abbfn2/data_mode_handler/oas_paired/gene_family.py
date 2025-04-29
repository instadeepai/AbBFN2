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
from abbfn2.data_mode_handler.oas_paired.constants import VALID_GENE_FAMILIES
from abbfn2.data_mode_handler.utils import load_from_hdf5, write_to_hdf5


class GeneFamilyHandler(DataModeHandler):
    """Base class for germline gene family data handling."""

    def __init__(self, dm_key: str, unknown_label: str = "unknown"):
        """Initialise the DM handler with dicts to convert gene families to classes.

        Args:
            dm_key (str): The key by which to identify the gene family type to be
                processed.
            unknown_label (str): The label to be used for missing gene families.
                Defaults to "unknown".
        """
        self.dm_key = dm_key
        self.genes_list = list(VALID_GENE_FAMILIES[self.dm_key].keys())
        self.gene2id = {gene: idx for idx, gene in enumerate(self.genes_list)}

        self.id2gene = {idx: k for k, idx in self.gene2id.items()}

        # Define an "unknown" id for genes not in the valid list.
        self.unknown_id = len(self.gene2id)
        self.unknown_label = unknown_label
        self.gene2id[self.unknown_label] = self.unknown_id
        self.id2gene[self.unknown_id] = self.unknown_label

    def sample_to_data(self, sample: Array) -> list[str]:
        """Converts an array of sample indices to a list of gene family labels.

        This method takes an array of indices and maps each index to its corresponding
        gene call label as defined in the id2gene dictionary attribute of the class.
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
            List[int]: A list of gene family labels corresponding to the input sample
                indices.
        """
        inp_shp = sample.shape
        if sample.ndim > 1:
            sample = sample.squeeze()

        if sample.ndim == 0:
            sample = sample[None]
        elif sample.ndim > 1:
            raise ValueError(f"Sample has unexpected shape: {inp_shp}")

        # Convert to gene family labels.
        return [self.id2gene.get(int(idx), self.unknown_label) for idx in sample]

    def data_to_sample(self, data: list[str]) -> Array:
        """Converts a list of gene family labels to a numpy array of sample indices.

        Args:
            data: A list of gene family labels.

        Returns:
            Array: A numpy array of sample indices.
        """
        return np.array([[self.gene2id.get(lab, self.unknown_id)] for lab in data])

    def save_data(
        self,
        data: np.ndarray,
        dir_path: Path,
        name: str = "properties.hdf5",
        exists_ok: bool = True,
    ) -> None:
        """Saves data to an hdf5 file."""
        path = dir_path / name
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")
        logging.info(f"Saving {self.dm_key} family labels to {path}.")
        data = {f"{self.dm_key}_family": data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> np.ndarray:
        """Loads data."""
        path = dir_path / name
        data = load_from_hdf5(path, keys=[f"{self.dm_key}_family"])
        return data[f"{self.dm_key}_family"]
