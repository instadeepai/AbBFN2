from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from numpy.typing import ArrayLike

from abbfn2.data.types import PreprocessFunction


class DataModeHandler(ABC):
    """Abstract base class for handling different data modes.

    Defines the interface for data mode handlers, which manage the preprocessing and preparation
    of data for training or evaluation. This includes defining dataset transformations and preparing
    ground truth and mask arrays outside of the network functions.

    Each subclass should implement the methods to handle specific data modes according to the
    requirements of different parts of the pipeline.
    """

    @abstractmethod
    def get_preprocess_function(
        self,
    ) -> tuple[PreprocessFunction, float]:
        """Retrieves the preprocessing function for the data mode, and it's associated meta-data.

        The preprocessing function should takes a raw batch (i.e. from the dataloader) and returns
        the batch information for this data mode and the float determines the priority with which this preprocessing
        function should be applied (lower values are applied first).

        DataModeBatch.x: A batch of data, where keys and values should conform to the specific requirements
                         of the data mode handler (e.g., including 'tokens' for tokenized text data).
        DataModeBatch.mask: A mask array where each element indicates the relevance of the corresponding data point
                           in the batch for evaluation (e.g., 1 for relevant, 0 for irrelevant).
                           The shape and data type of the array should match those of the ground truth data.

        Returns:
            Tuple[PreprocessFunction, bool, float]: The preprocessing function and the priority of the function.
        """
        pass

    def sample_to_mask(self, sample: ArrayLike) -> ArrayLike:
        """Infers a mask from a generated sample.

        This function is primarily used to generate a mask after the BFN has prepared a sample.

        Args:
            sample (ArrayLike): A sample for the data mode.

        Returns:
            ArrayLike: An inferred mask.  By default the mask is all ones and the shape of the sample (i.e. nothing is masked).
        """
        return jnp.ones_like(sample)

    @abstractmethod
    def sample_to_data(self, sample: ArrayLike) -> Any:
        """Converts a sample(s) to a canoncical data format for the mode.

        Args:
            sample (ArrayLike): A sample for the data mode.

        Returns:
            Any: The sample in a canonical data format for the mode.
        """
        pass

    @abstractmethod
    def data_to_sample(self, data: Any) -> ArrayLike:
        """Converts data to a sample for the mode.

        Args:
            data (Any): The data to convert to a sample.

        Returns:
            ArrayLike: The sample for the mode.
        """
        pass

    @abstractmethod
    def save_data(
        self,
        data: Any,
        dir_path: Path,
        name: str,
        exists_ok: bool = True,
    ) -> None:
        """Saves data to a file.

        Args:
            data (Any): The data to save.
            dir_path (Path): The directory to save the data to.
            path (str): The (typically file) name to save the data with.
            exists_ok (bool): Whether it is ok if the file already exists.
        """
        pass

    @abstractmethod
    def load_data(self, dir_path: Path, name: str) -> Any:
        """Loads data from a file.

        Args:
            dir_path (Path): The directory in which the data is stored.
            path (str): The (typically file) name to load the data from.

        Returns:
            Any: The loaded data.
        """
        pass
