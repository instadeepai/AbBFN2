from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from abbfn2.data.data_mode_handler.base import DataModeHandler
from abbfn2.data.types import DataModeBatch, RawBatch


def preprocess_sequence_length(
    raw_batch: RawBatch,
) -> tuple[DataModeBatch, RawBatch]:
    """Computes the length of each sequence in a batch and encapsulates the lengths in a DataModeBatch.

    Args:
        raw_batch: Batch of raw data containing sequences.

    Returns:
        A tuple containing the DataModeBatch with sequence lengths and a mask of ones, and the unchanged raw_batch.
    """
    # Get sequence lengths. Note: SequenceDataModeHandler is assumed to have been applied before this handler, so the
    # "sequence" key is already present.
    sequence_lengths = np.array([len(seq) for seq in raw_batch["sequence"]])
    sequence_lengths = sequence_lengths[..., None]  # [N] --> [N, 1]

    dm_batch = DataModeBatch(
        x=sequence_lengths,
        mask=np.ones_like(sequence_lengths),
    )

    return dm_batch, raw_batch


class SequenceLengthDataModeHandler(DataModeHandler):
    """Handles data mode specific to sequence length.

    This class extends `DataModeHandler` to implement methods for handling datasets where the primary
    feature is the length of sequences. It includes a method for adding sequence length to the dataset,
    preparing ground truth data based on sequence lengths, and generating a mask where all elements are
    considered relevant.
    """

    def get_preprocess_function(
        self,
    ) -> tuple[Callable[[RawBatch], DataModeBatch], bool, float]:
        """Defines and returns the preprocessing functions for sequence length data.

        Returns:
            Tuple[Callable[[RawBatch], DataModeBatch], bool, float]: The preprocessing function, whether it is batched,
            and the priority of the function.
        """
        return preprocess_sequence_length, 1.0

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
        """TODO"""  # noqa: D415
        return np.array(data)

    def load_data(self, path: Path) -> Any:
        """TODO"""  # noqa: D415
        if path.is_dir():
            path = path / "sequence_lengths.npy"
        return np.load(path)
