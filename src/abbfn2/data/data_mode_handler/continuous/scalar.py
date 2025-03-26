import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from  abbfn2.data.data_mode_handler.base import DataModeHandler
from  abbfn2.data.data_mode_handler.utils import load_from_hdf5, write_to_hdf5
from abbfn2.data.types import DataModeBatch, RawBatch
from jax import Array
from numpy import ndarray

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


def preprocess_scalar_values(
    raw_batch: RawBatch,
    carry_args: dict,
    dm_key: str,
    data_bounds: tuple[int | float, int | float],
    target_bounds: tuple[int | float, int | float] | None = (-1.0, 1.0),
    unknown_value: int | float = 100.0,
) -> tuple[DataModeBatch, RawBatch]:
    """Preprocesses a scalar values data mode.

    Missing values are first replaced with an unknown value. Then, data are scaled to a
    target range, and reshaped to fit the continuous BFN.

    Args:
        raw_batch (RawBatch): The raw input batch.
        carry_args (dict): A dictionary to store any arguments that persist between data
            modes. This can be used or modified and will be "seen" by subsequent
            preprocessing functions (based on their priority in the DataModeHandler).
        dm_key (str): The key to identify data in the raw batch.
        src_bounds (tuple[int | float, int| float]): Tuple of lower and upper bounds of
            the INPUT range.
        trg_bounds (tuple[int | float, int| float] | None): Tuple of lower and upper bounds of
            the OUTPUT range. Defaults to (-1,1). If None, no scaling will be applied.
        unknown_value (int | float, optional): The value with which to fill missing
            data. Defaults to 100.0.

    Returns:
        tuple[DataModeBatch, RawBatch, dict]: The preprocessed data, the raw batch, and
            the carry args.
    """
    data_min, data_max = data_bounds

    # SQL data format. Replace SQL NULLs with the impute value.
    data = np.array(
        [
            e if (isinstance(e, float | int)) else unknown_value
            for e in raw_batch[dm_key]
        ]
    )

    # Clip to the specified range
    data = np.clip(data, data_min, data_max)

    # If a target bound is specified, scale the data.
    if target_bounds:
        data = scale_values(data, src_bounds=data_bounds, trg_bounds=target_bounds)

    data = data.reshape((len(raw_batch[dm_key]), 1))  # [batch_size, 1]

    dm_batch = DataModeBatch(
        x=data,
        mask=np.ones_like(data[:, :]),
    )

    return dm_batch, raw_batch, carry_args


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

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], bool, float]:
        """Defines and returns the preprocessing functions for scalar data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], bool, float]: The preprocessing
                function, whether it is batched, and the priority of the function.
        """
        # Note: The preprocess function is defined external to this class such that it can be pickled for
        # multiprocessing. This is necessary for efficient data loading and preprocessing.
        preprocess_fn = partial(
            preprocess_scalar_values,
            dm_key=self.dm_key,
            data_bounds=self.data_bounds,
            target_bounds=self.target_bounds,
            unknown_value=self.unknown_value,
        )

        priority = 1.0

        return preprocess_fn, priority

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
