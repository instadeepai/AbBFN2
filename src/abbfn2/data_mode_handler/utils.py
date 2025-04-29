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

import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from abbfn2.data_mode_handler import DataModeHandler


def save_samples(
    samples: dict[str, Any],
    dm_handlers: dict[str, DataModeHandler],
    path: Path | str,
    save_handlers: bool | None = True,
) -> None:
    """Saves samples to a specified directory using the appropriate data mode handlers.

    Saves samples to a specified directory using the appropriate data mode handlers. Each sample, typically an output
    from a network in a possibly tokenized or encoded form, is converted into a canonical form for the corresponding
    data mode (e.g., a protein sequence from integers representing tokenized amino acids to an amino acid sequence
    saved in a FASTA file) before being saved.

    Args:
        samples: A dictionary where keys are data modes and values are samples to be saved, typically in a format specific to the network's output.
        dm_handlers: A dictionary mapping data modes to their respective DataModeHandler objects, responsible for
                     transforming samples to canonical data forms and saving them.
        path: The directory path where samples should be saved. Can be a string or Path object.

    Raises:
        ValueError: If the specified path is not a directory.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Specified path {path} is not a directory.")

    for dm, sample in samples.items():
        dm_handler = dm_handlers[dm]
        data = dm_handler.sample_to_data(
            sample,
        )  # Transform sample to canonical data form.
        dm_handler.save_data(data, dir_path=path)  # Save the canonical data.

    if save_handlers:
        with open(path / "dm_handlers.pkl", "wb") as f:
            pickle.dump(dm_handlers, f)  # Serialize and save the dm_handlers dictionary


def load_samples(
    path: Path | str,
    dm_handlers: dict[str, DataModeHandler] | None = None,
    return_data: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Loads samples from a specified directory.

    Loads samples from a specified directory, optionally loading DataModeHandler objects from a pickle file if not
    explicitly passed. This function transforms the canonical data form back into the format used by the network.
    If `dm_handlers` is not provided, the function attempts to load it from `dm_handlers.pkl` in the specified directory.

    Args:
        path: The directory path from which samples should be loaded. Can be a string or Path object.
        dm_handlers: Optional; a dictionary mapping data modes to their respective DataModeHandler objects. If None,
                     attempts to load them from `dm_handlers.pkl` within the specified path.
        return_data: If True, returns a tuple containing both raw data and samples, providing insight into both the
                     canonical and network-specific representations of the data.

    Returns:
        A dictionary of samples or a tuple of raw data and samples, depending on the value of return_data.

    Raises:
        ValueError: If the specified path is not a directory or if dm_handlers cannot be loaded from the path.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Specified path {path} is not a directory.")

    if dm_handlers is None:
        handlers_path = path / "dm_handlers.pkl"
        if not handlers_path.exists():
            raise ValueError(f"DataModeHandler's not found at {handlers_path}.")
        with open(handlers_path, "rb") as f:
            dm_handlers = pickle.load(f)

    data, samples = {}, {}
    for dm, handler in dm_handlers.items():
        data_dm = handler.load_data(path)  # Load the canonical data.
        sample_dm = handler.data_to_sample(
            data_dm,
        )  # Transform canonical data to sample form.
        data[dm], samples[dm] = data_dm, sample_dm

    if return_data:
        return samples, data
    else:
        return samples


def load_data(
    path: Path | str,
    dm_handlers: dict[str, DataModeHandler] | None = None,
) -> dict[str, Any]:
    """Loads data using the provided data mode handlers.

    Args:
        path: The directory path from which data should be loaded. Can be a string or Path object.
        dm_handlers: A dictionary mapping data modes to their respective DataModeHandler objects.

    Returns:
        A dictionary where keys are data modes and values are the data loaded for each mode.

    Raises:
        ValueError: If the specified path is not a directory.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Specified path {path} is not a directory.")

    if dm_handlers is None:
        handlers_path = path / "dm_handlers.pkl"
        if not handlers_path.exists():
            raise ValueError(f"DataModeHandler's not found at {handlers_path}.")
        with open(handlers_path, "rb") as f:
            dm_handlers = pickle.load(f)

    data = {}
    for dm, handler in dm_handlers.items():
        data[dm] = handler.load_data(path)

    return data


def write_to_hdf5(
    path: str,
    data: dict[str, np.ndarray | str | list[str]],
    delete_if_exists: bool = False,
) -> None:
    """Writes data to an HDF5 file, supporting numeric arrays and strings.

    Args:
        path (str): The file path to write the HDF5 data.
        data (Dict[str, Union[np.ndarray, str, List[str]]]): A dictionary containing the data to write.
            Keys are used as dataset names. Values can be numeric arrays or strings (single or list of strings).
        delete_if_exists (bool): If True, the file at `path` will be overwritten;
            if False, new data will be appended to existing file, with existing datasets being replaced.

    Raises:
        ValueError: If the value type is not supported (not a numpy array or string).
    """
    mode = "w" if delete_if_exists else "a"
    with h5py.File(path, mode) as file:
        for k, v in data.items():
            if not delete_if_exists and k in file:
                del file[k]
            if isinstance(
                v,
                np.ndarray | list,
            ):  # Handle numeric arrays and lists (for strings)
                # Determine if the data is a list of strings
                if isinstance(v, list) and all(isinstance(item, str) for item in v):
                    dt = h5py.string_dtype(encoding="utf-8")
                    dataset = file.create_dataset(k, (len(v),), dtype=dt)
                    dataset[:] = v
                else:
                    file.create_dataset(k, data=v)
            elif isinstance(v, str):  # Handle single strings
                dt = h5py.string_dtype(encoding="utf-8")
                dataset = file.create_dataset(k, (1,), dtype=dt)
                dataset[0] = v
            else:
                raise ValueError(f"Unsupported value type for key {k}: {type(v)}")


def load_from_hdf5(
    path: str,
    keys: str | list[str] | None = None,
) -> dict[str, Any]:
    """Loads data from an HDF5 file.

    Loads data from an HDF5 file, automatically decoding byte strings to UTF-8 strings.
    Allows for a single key, a list of keys, or None to be specified for loading specific datasets.

    Args:
        path (str): The file path from which to load HDF5 data.
        keys (Optional[Union[str, List[str]]]): A single dataset name, a list of dataset names to load,
                                                or None to load all datasets in the file.

    Returns:
        Dict[str, Any]: A dictionary with keys corresponding to dataset names and values to the data loaded,
                        with byte strings decoded to UTF-8 strings.
    """
    data = {}
    with h5py.File(path, "r") as file:
        if keys is None:
            keys_to_load = file.keys()
        elif isinstance(keys, str):
            keys_to_load = [keys]
        elif isinstance(keys, list):
            keys_to_load = keys
        else:
            raise ValueError("keys must be None, a string, or a list of strings.")

        for k in keys_to_load:
            if k in file:
                dataset = file[k]
                if dataset.dtype == h5py.string_dtype():
                    # Handle variable-length string datasets
                    val = (
                        [str(s, "utf-8") for s in dataset[:]]
                        if dataset.len() > 1
                        else str(dataset[0], "utf-8")
                    )
                else:
                    val = dataset[()]

                # Convert a numpy array of objects (bytes) to strings, if applicable
                if isinstance(val, np.ndarray) and val.dtype == object:
                    val = np.array(val.tolist(), dtype=str)

                data[k] = val
            else:
                data[k] = None  # or raise an error, or continue based on preference

    return data
