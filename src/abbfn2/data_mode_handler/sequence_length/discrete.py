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

from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from abbfn2.data_mode_handler.base import DataModeHandler


class SequenceLengthDataModeHandler(DataModeHandler):
    """Handles data mode specific to sequence length.

    This class extends `DataModeHandler` to implement methods for handling datasets where the primary
    feature is the length of sequences. It includes a method for adding sequence length to the dataset,
    preparing ground truth data based on sequence lengths, and generating a mask where all elements are
    considered relevant.
    """

    def sample_to_data(self, sample: Array) -> Array:
        """Converts a sample to data.

        Args:
            sample (Array): An array of sequence lengths.

        Returns:
            Array: The input array of sequence lengths.
        """
        return sample

    def save_data(
        self,
        data: Any,
        out_dir: Path,
        name: str = "sequence_lengths.npy",
        exists_ok: bool = True,
    ) -> None:
        """Saves a set of data."""
        path = out_dir / name

        # Check if the file exists and handle according to exists_ok flag
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")

        np.save(path, data)

    def data_to_sample(self, data: Array) -> Array:
        return np.array(data)

    def load_data(self, path: Path) -> Any:
        if path.is_dir():
            path = path / "sequence_lengths.npy"
        return np.load(path)
