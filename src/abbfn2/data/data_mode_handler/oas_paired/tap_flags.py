import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.data_mode_handler.utils import load_from_hdf5, write_to_hdf5
from abbfn2.data.types import DataModeBatch, RawBatch
from jax import Array

from abbfn2.data.data_mode_handler.oas_paired.constants import TAP_REFERENCE_VALUES


def get_tap_flag(
    value: int | float | None, metric: str, unknown_value: str = "unknown"
) -> str:
    """Assigns a flag based on reference values given a TAP metric and value.

    Args:
        value (int | float | None): The value based on which the flag will be
            calculated.
        metric (str): The TAP metric whose reference values will be used.
        unknown_value (str): The unknown flag assigned to values that cannot
            be assigned a valid flag. Defaults to "unknown".

    Raises:
        ValueError: If the provided metric is invalid.

    Returns:
        str: The assigned TAP flag.
    """
    if not isinstance(value, int | float):
        return unknown_value

    refs = TAP_REFERENCE_VALUES

    if metric not in refs:
        raise ValueError(f"{metric} is not a valid TAP metric.")

    thresholds = refs[metric]

    match metric:
        case "psh":
            if thresholds[0.05] <= value < thresholds[0.95]:
                return "green"
            elif thresholds[0.0] <= value <= thresholds[1.0]:
                return "amber"
            else:
                return "red"
        case "ppc" | "pnc":
            if value <= thresholds[0.95]:
                return "green"
            elif value <= thresholds[1.0]:
                return "amber"
            else:
                return "red"
        case "sfvcsp":
            if value >= thresholds[0.05]:
                return "green"
            elif value >= thresholds[0.0]:
                return "amber"
            else:
                return "red"


def preprocess_tap_flags(
    raw_batch: RawBatch,
    carry_args: dict,
    metric: str,
    flag2id: dict[str, int],
    unknown_id: int,
) -> tuple[DataModeBatch, RawBatch, dict]:
    """Preprocesses the TAP values into discrete flags.

    Args:
        raw_batch (RawBatch): The raw batch of data.
        carry_args (dict): A dictionary to store any arguments that persist between data
            modes. This can be used or modified and will be "seen" by subsequent
            preprocessing functions (based on their priority in the DataModeHandler).
        dm_key (str): The key to identify data in the raw batch.
        flag2id (dict[str, int]): A dictionary mapping flags to ids.
        unknown_id (int): The index for unknown flags.

    Returns:
        Tuple[DataModeBatch, RawBatch]: The preprocessed data and the raw batch.
    """
    flags = [get_tap_flag(val, metric) for val in raw_batch[metric]]
    flag_ids = np.array([flag2id.get(flag, unknown_id) for flag in flags])
    flag_ids = flag_ids[..., None]  # [N, 1]

    dm_batch = DataModeBatch(
        x=flag_ids,
        mask=np.ones_like(flag_ids),
    )

    return dm_batch, raw_batch, carry_args


class TapFlagsHandler(DataModeHandler):
    """Base class for discrete TAP flags."""

    def __init__(self, metric: str, unknown_label: str = "unknown"):
        """Initialise class with dictionaries for conversions to categories."""
        self.metric = metric
        self.dm_key = self.metric + "_flag"

        self.flag2id = {"green": 0, "amber": 1, "red": 2}
        self.id2flag = {idx: k for k, idx in self.flag2id.items()}
        self.unknown_id = len(self.flag2id)
        self.unknown_label = unknown_label
        self.flag2id[self.unknown_label] = self.unknown_id
        self.id2flag[self.unknown_id] = self.unknown_label

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], float]:
        """Defines and returns the preprocessing functions for TAP flags.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], float]: The preprocessing
                function and the priority of the function.
        """
        preprocess_fn = partial(
            preprocess_tap_flags,
            metric=self.metric,
            flag2id=self.flag2id,
            unknown_id=self.unknown_id,
        )

        priority = 1.0
        return preprocess_fn, priority

    def sample_to_data(self, sample: Array) -> list[str]:
        """Converts an array of sample indices to a list of TAP flag labels.

        This method takes an array of indices and maps each index to its corresponding
        TAP flag label as defined in the id2flag dictionary attribute of the class.
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
        inp_shp = sample.shape
        if sample.ndim > 1:
            sample = sample.squeeze()

        if sample.ndim == 0:
            sample = sample[None]
        elif sample.ndim > 1:
            raise ValueError(f"Sample has unexpected shape: {inp_shp}")

        # Convert to length labels.
        return [self.id2flag.get(int(idx), self.unknown_label) for idx in sample]

    def data_to_sample(self, data: list[str]) -> Array:
        """Converts a list of TAP flag labels to a numpy array of sample indices.

        Args:
            data: A list of TAP flag labels.

        Returns:
            Array: A numpy array of sample indices.
        """
        return np.array([[self.flag2id.get(lab, self.unknown_id)] for lab in data])

    def save_data(
        self,
        data: np.ndarray,
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
        if path.exists() and not exists_ok:
            raise FileExistsError(f"The file {path} already exists.")
        logging.info(f"Saving {self.dm_key} labels to {path}.")
        data = {self.dm_key: data}
        write_to_hdf5(path, data, delete_if_exists=not exists_ok)

    def load_data(self, dir_path: Path, name: str = "properties.hdf5") -> np.ndarray:
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
