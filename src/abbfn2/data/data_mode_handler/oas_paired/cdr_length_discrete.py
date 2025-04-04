import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from jax import Array

from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.data_mode_handler.oas_paired.constants import (
    VALID_REGION_LENGTHS as VALID_CDR_LENGTHS,
)
from abbfn2.data.data_mode_handler.utils import load_from_hdf5, write_to_hdf5
from abbfn2.data.types import DataModeBatch, RawBatch


def preprocess_cdr_lengths(
    raw_batch: RawBatch,
    carry_args: dict,
    dm_key: str,
    unknown_id: int,
    len2id: dict[int, int],
) -> tuple[DataModeBatch, RawBatch, dict]:
    """Preprocesses the CDR lengths data by mapping lengths to their respective IDs.

    Args:
        raw_batch (RawBatch): A batch of raw data, expected to include 'dm_key' key.
        carry_args (dict):  A dictionary to store any arguments that persist between
            data modes.
        dm_key (str): The key by which to identify the CDR loop to be processed.
        len2id (dict[int, int]): A dictiionary mapping the length of CDRs to unique IDs.
        unknown_label (int): The ID to be used for missing CDR lengths.
            Defaults to -1.

    Returns:
        tuple[DataModeBatch, RawBatch, carry_args]: A tuple containing a `DataModeBatch`
            object with processed CDR IDs and a mask, the unchanged input `raw_batch`,
            and the carry_args dictionary.
    """
    length_ids = np.array(
        [len2id.get(cdr_len, unknown_id) for cdr_len in raw_batch[dm_key]], dtype=int
    )  # [N]
    length_ids = length_ids[..., None]  # [N, 1]

    dm_batch = DataModeBatch(
        x=length_ids,
        mask=np.ones_like(length_ids),
    )

    return dm_batch, raw_batch, carry_args


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

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], bool, float]:
        """Defines and returns the preprocessing functions for CDR length data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], float]: The preprocessing
                function and the priority of the function.
        """
        preprocess_fn = partial(
            preprocess_cdr_lengths,
            dm_key=self.dm_key,
            unknown_id=self.unknown_id,
            len2id=self.len2id,
        )

        priority = 1.0
        return preprocess_fn, priority

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
