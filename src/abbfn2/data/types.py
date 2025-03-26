from collections.abc import Callable
from typing import Any, NamedTuple

from numpy.typing import ArrayLike

RawBatch = dict[str, list[Any]]


class DataModeBatch(NamedTuple):
    """A named tuple representing a batch of data in a specific data mode.

    Attributes:
        x (ArrayLike): The input data.
        mask (ArrayLike): The mask for the input data.
    """

    x: ArrayLike
    mask: ArrayLike


class Batch(NamedTuple):
    """A named tuple representing a batch of data.

    Attributes:
        x (Dict[str, ArrayLike]): The input data.
        mask (Dict[str, ArrayLike]): The mask for the input data.
    """

    x: dict[str, ArrayLike]
    mask: dict[str, ArrayLike]


PreprocessFunction = Callable[[RawBatch, dict], tuple[DataModeBatch, RawBatch, dict]]
