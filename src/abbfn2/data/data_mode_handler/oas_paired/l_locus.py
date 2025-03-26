import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.data_mode_handler.utils import load_from_hdf5, write_to_hdf5
from abbfn2.data.types import DataModeBatch, RawBatch
from jax import Array


def preprocess_l_loci(
    raw_batch: RawBatch,
    carry_args: dict,
    dm_key: str,
    locus2id: dict[str, int],
    unknown_id: int,
) -> tuple[DataModeBatch, RawBatch, dict]:
    """Preprocesses the light chain locus data by mapping locus labels to their corresponding IDs.

    Args:
        raw_batch (RawBatch): A batch of raw data, expected to include 'locus' labels.
        carry_args (dict):  A dictionary to store any arguments that persist between
            data modes.
        dm_key (str): The key by which to identify the light chain locus.
        locus2id (Dict[str, int]): A dictionary mapping locus names (str) to unique IDs (int).
        unknown_id (int): The ID to be used for loci that are not found in the `locus2id` mapping.

    Returns:
        Tuple[DataModeBatch, RawBatch]: A tuple containing a `DataModeBatch` object with
            processed locus IDs and a mask, and the unchanged input `raw_batch`.
    """
    locus_ids = np.array(
        [locus2id.get(lab, unknown_id) for lab in raw_batch[dm_key]],
    )

    locus_ids = locus_ids[..., None]  # [N] --> [N, 1]

    dm_batch = DataModeBatch(
        x=locus_ids,
        mask=np.ones_like(locus_ids),
    )

    return dm_batch, raw_batch, carry_args


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

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], float]:
        """Defines and returns the preprocessing functions for locus data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], float]: The preprocessing
                function and the priority of the function.
        """
        # Note: The preprocess function is defined external to this class such that it can be pickled for
        # multiprocessing. This is necessary for efficient data loading and preprocessing.
        preprocess_fn = partial(
            preprocess_l_loci,
            dm_key=self.dm_key,
            locus2id=self.locus2id,
            unknown_id=self.unknown_id,
        )
        priority = 1.0

        return preprocess_fn, priority

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
