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
import pickle
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from huggingface_hub import hf_hub_download
from jax import Array
from omegaconf import DictConfig
from tabulate import tabulate

from abbfn2.bfn import BFN, ContinuousBFN, DiscreteBFN, MultimodalBFN
from abbfn2.bfn.types import (
    OutputNetworkPrediction,
    OutputNetworkPredictionContinuous,
    OutputNetworkPredictionDiscrete,
)
from abbfn2.data_mode_handler import DataModeHandler, load_samples
from abbfn2.data_mode_handler.oas_paired.constants import IMGT2IDX


def show_conditioning_settings(num_samples, samples, masks):
    log_str = f"Loaded {num_samples} samples and masks:"
    get_gen_str = lambda m: (  # noqa
        "Condition" if m.all() else "Predict" if (1 - m).all() else "Mixed"
    )
    tb_data = [
        (dm, str(samples[dm].shape), str(masks[dm].shape), get_gen_str(masks[dm]))
        for dm in samples.keys()
    ]
    tab_str = tabulate(
        tb_data,
        headers=["DataMode", "x (shape)", "m (shape)", "Gen. mode"],
        tablefmt="rounded_outline",
    )
    log_str += "\n\t" + tab_str.replace("\n", "\n\t")
    logging.info(log_str)


def flatten_and_crop(x, inputs_info=None):
    nb, bs, *dims = x.shape
    trg_shp = [
        nb * bs,
    ] + dims
    x = x.reshape(*trg_shp)
    return x[: inputs_info["num_samples"]] if inputs_info is not None else x


def pad_and_reshape(x, num_samples_padded, num_batches):
    """Pads the input array to a specified size and reshapes it for batch processing across multiple devices.

    Args:
        x (np.ndarray): The input array with shape [num_samples, ...].

    Returns:
        np.ndarray: The reshaped array with shape [num_batches, batch_size, <remaining_dims>].

    Raises:
        ValueError: If `num_samples_padded` is less than the number of samples in `x`.
    """
    if num_samples_padded < x.shape[0]:
        raise ValueError(
            f"`num_samples_padded` (={num_samples_padded}) must be greater than or equal to the number of samples in `x` (={x.shape[0]}).",
        )

    x = np.pad(
        x,
        [(0, num_samples_padded - x.shape[0])] + (x.ndim - 1) * [(0, 0)],
        mode="constant",
        constant_values=0,
    )
    # Reshape [num_samples_padded, <remaining_dims>] -> [num_batches, batch_size, <remaining_dims>]
    trg_shp = [num_batches, -1] + list(x.shape[1:])
    x = x.reshape(*trg_shp)
    return x


def get_input_samples(
    cfg: DictConfig,
    bfn: MultimodalBFN,
    dm_handlers: dict[str, DataModeHandler],
    key: random.PRNGKey,
) -> dict[str, Array]:
    """Retrieves input samples based on configuration settings and data mode handlers.

    This function supports loading input samples from a specified path
    or generating them based on provided Bayesian Functional Network (BFN) configurations.
    When loading from a path, it ensures the loaded samples match the requested data modes (DMs),
    and can filter or adjust samples based on the number of samples specified in the configuration.
    If samples are generated, it relies on the types of BFN (Continuous, Discrete)
    to create appropriate prior predictions. Additionally, it handles overwrites for specific DMs
    as per configuration.

    Args:
        cfg: A DictConfig object containing configuration settings such as path, num_input_samples,
             num_samples, and dm_overwrites. The configuration dictates how samples are loaded or generated.
        bfn: A MultimodalBFN instance used for generating samples when not loading from a path.
             The BFN's configuration determines the shape and type of the generated samples.
        dm_handlers: A dictionary mapping data mode names to DataModeHandler instances. These handlers
                     are used to convert data into samples or adjust samples to match specific requirements.
        key: A JAX PRNGKey used for random number generation in sample creation.

    Returns:
        A dictionary where keys are data mode names and values are arrays of samples. The structure
        and content of the returned samples depend on the input cfg and the types of BFN involved.

    Raises:
        AssertionError: If samples are not loaded from disk and num_input_samples is not explicitly set.
        Exception: If the requested DataModes are missing from the loaded samples or if an unsupported
                   BFN type is encountered during sample generation.
    """
    if cfg.path is not None:
        # Load samples from file.
        input_path = cfg.path
        logging.info(f"Loading samples from: {input_path}.")
        samples = load_samples(input_path, dm_handlers)

        # Count number of loaded samples and select first "num_samples" if needed.
        num_loaded = samples[list(samples.keys())[0]].shape[0]
        if cfg.num_input_samples is not None and num_loaded > cfg.num_input_samples:
            samples = jax.tree_util.tree_map(
                lambda x: x[: cfg.num_input_samples],
                samples,
            )

        # Check if, and handle if needed, the requested DataModes match those in the loaded sample.
        sample_dms = set(samples.keys())
        target_dms = set(dm_handlers.keys())

        if sample_dms != target_dms:
            # Compute list of dm
            sample_only_dms = sample_dms - target_dms
            target_only_dms = target_dms - sample_dms
            if len(target_only_dms) > 0:
                # We can't load dm's that are missing in samples; raise an error.
                raise Exception(
                    f"Requested DataModes are missing from the samples: {target_only_dms}.",
                )
            if len(sample_only_dms) > 0:
                # We can ignore dm's that exist in the samples but are not requested.
                samples = {dm: samples[dm] for dm in target_dms}
                logging.info(
                    f"The following data modes are present in the loaded samples but unused: {sample_only_dms}",
                )

    else:
        assert (
            cfg.num_input_samples is not None
        ), "If samples are not loaded from disk, num_input_samples must be explicitly set."

        def get_prior_pred(bfn: BFN) -> OutputNetworkPrediction:
            """Generates a prior prediction for a given Bayesian Functional Network (BFN) based on its type.

            This function supports ContinuousBFN and DiscreteBFN creating a default prior
            prediction for each. The prior prediction is then used to sample initial values for the network's
            input variables. For continuous BFNs, the prediction includes means and variances,
            while for discrete BFNs, it comprises logits representing class probabilities.

            Args:
                bfn: An instance of BFN (Bayesian Functional Network), which can be of type ContinuousBFN,
                     DiscreteBFN. The specific type of BFN determines the structure and
                     content of the generated prior prediction.

            Returns:
                An instance of one of the following, depending on the BFN type:
                - OutputNetworkPredictionContinuous: For ContinuousBFN, containing zero means and unit variances.
                - OutputNetworkPredictionDiscrete: For DiscreteBFN, containing logits of ones.
                  and the specified number of bins.

            Raises:
                Exception: If the BFN type is unsupported, indicating a programming error or an unexpected BFN configuration.
            """
            if isinstance(bfn, ContinuousBFN):
                pred = OutputNetworkPredictionContinuous(
                    x=jnp.zeros(bfn.cfg.variables_shape),
                    rho=jnp.ones(bfn.cfg.variables_shape),
                )
            elif isinstance(bfn, DiscreteBFN):
                pred = OutputNetworkPredictionDiscrete(
                    logits=jnp.ones(bfn.cfg.variables_shape + (bfn.cfg.num_classes,)),
                )

            else:
                raise Exception(f"Unsupported BFN type: {type(bfn)}.")

            return pred

        # Prepare a num_samples random samples for each data mode.
        # samples: {dm: [num_samples, ......var_shape...], ...}
        samples = jax.tree_util.tree_map(
            lambda bfn: get_prior_pred(bfn)
            .to_distribution()
            .sample(seed=key, sample_shape=(cfg.num_input_samples,)),
            bfn.bfns,
        )

    # If dm_overwrites are provided, apply them to the samples.
    if "dm_overwrites" in cfg and cfg.dm_overwrites is not None:
        for dm, dm_data in cfg.dm_overwrites.items():
            logging.info(f"Setting '{dm}' to {dm_data} for all input samples.")
            if not isinstance(dm_data, list):
                dm_data = [dm_data]
            dm_data = np.array(dm_data)
            dm_sample = dm_handlers[dm].data_to_sample(dm_data)
            dm_sample = dm_sample.squeeze()
            samples[dm] = np.broadcast_to(dm_sample, samples[dm].shape)
    return samples


def configure_imgt_position_overrides(
    masks: dict[str, Array | np.ndarray],
    samples: dict[str, Array | np.ndarray],
    overrides: dict[tuple[str, tuple[int, str]], str],
    dm_handlers: dict[str, DataModeHandler],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Configure position-specific overrides for IMGT-based sequence regions.

    This function updates the `samples` and `masks` dictionaries based on the
    given overrides, mapping positions and residues to their corresponding
    indices within specified sequence regions.

    Args:
        masks (dict[str, np.ndarray]): A dictionary of binary masks indicating valid positions
                                       within sequence regions, keyed by region names. Can contain
                                       either `jax.numpy` or `numpy` arrays.
        samples (dict[str, np.ndarray]): A dictionary of sequence data arrays, keyed by region names.
                                         Can contain either `jax.numpy` or `numpy` arrays.
        overrides (dict[tuple[str, tuple[int, str]], str]): A dictionary of overrides. Keys are tuples
                                                            of chain, position, and residue, and values
                                                            are the desired sequence values.
        dm_handlers (dict[str, DataModeHandler]): Handlers for managing data modes and tokenizers,
                                                  keyed by region names.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: Updated `samples` and `masks` dictionaries with
                                                             position-specific overrides applied.

    Example:
        ```
        masks = {"region1_seq": np.zeros((10, 100))}
        samples = {"region1_seq": np.zeros((10, 100))}
        overrides = {("A", (15, "A")): "value"}
        dm_handlers = {"region1_seq": handler_object}

        updated_samples, updated_masks = configure_imgt_position_overrides(masks, samples, overrides, dm_handlers)
        ```
    """
    # Convert jax.numpy arrays to numpy arrays if necessary
    masks = {
        key: np.array(value) if isinstance(value, jnp.ndarray) else value
        for key, value in masks.items()
    }
    samples = {
        key: np.array(value) if isinstance(value, jnp.ndarray) else value
        for key, value in samples.items()
    }
    logging.info(
        "Position-specific overrides were detected. These residues will be set and made visible to the network:"
    )

    for (chain, pos), res in overrides.items():
        chain_mapping = IMGT2IDX.get(chain, {})

        region = None
        idx = None
        res_to_idx = None

        # Locate the region and position index for the override
        for region_name, pos_to_idx in chain_mapping.items():
            if pos in pos_to_idx:
                region = region_name
                idx = pos_to_idx[pos]
                res_to_idx = {
                    token: i
                    for i, token in enumerate(
                        dm_handlers[f"{region}_seq"].tokenizer.vocabulary
                    )
                }
                break

        if region is None or idx is None or res_to_idx is None:
            raise ValueError(
                f"Invalid override position {pos} for chain {chain.upper()}."
            )

        # Apply the override
        logging.info(f"Position: {pos} ({region}) -> Residue: {res}")
        samples[f"{region}_seq"][:, idx] = res_to_idx[res]
        masks[f"{region}_seq"][:, idx] = 1

    return samples, masks


def generate_random_mask_from_array_visible_pad(arr, frac_fill=0.5, exclusions=None):
    """Generate a mask (same shape as arr) where:
      - Each row has a% of non-exclusion positions set to 0 (chosen randomly).
      - All other positions are set to 1.

    Parameters
    arr : numpy.ndarray of shape (n, m)
        Integer array from which to derive the mask.
    frac_fill : float
        Percentage of valid positions (non 1,4,5) in each row to set to 0.

    Returns:
    mask : numpy.ndarray of shape (n, m)
        The resulting 0/1 mask.
    """
    # Initialize mask of all ones
    if exclusions is None:
        exclusions = [1, 4, 5]

    mask = np.ones_like(arr, dtype=int)
    n, m = arr.shape

    for i in range(n):
        # Identify columns where arr[i] is NOT in {1,4,5}
        valid_positions = np.where(~np.isin(arr[i], exclusions))[0]

        # Calculate how many of these valid positions should become 0
        num_zero = int(round(len(valid_positions) * frac_fill))

        # Choose those positions randomly and set them to 0 in the mask
        if num_zero > 0:
            zero_indices = np.random.choice(
                valid_positions, size=num_zero, replace=False
            )
            mask[i, zero_indices] = 0

    return mask


def configure_output_dir(
    cfg: DictConfig,
) -> Path:
    """Configure and return the output directories and paths.

    Args:
        cfg (DictConfig): The current configuration.

    Returns:
        Path: The local directory.
    """
    exp_name = cfg.exp_name
    if exp_name is None:
        exp_name = datetime.now().strftime("%d-%m-%y_%H-%M")
    local_output_dir = Path(cfg.local_validation_dir) / exp_name
    local_output_dir.mkdir(parents=True, exist_ok=cfg.overwrite_local_if_exists)

    if not cfg.overwrite_local_if_exists:
        assert (
            not local_output_dir.exists()
        ), f"Local output directory {local_output_dir} already exists."

    return local_output_dir


def load_params(cfg: DictConfig) -> dict[str, jax.Array]:
    """Load the parameters from the model weights path or from the Hugging Face Hub.

    Args:
        cfg (DictConfig): The configuration.

    Returns:
        dict[str, jax.Array]: The parameters.
    """
    if cfg.load_from_hf:
        file_path = hf_hub_download(
            repo_id="InstaDeepAI/AbBFN2", filename="model_params.pkl"
        )
        with open(file_path, "rb") as f:
            params = pickle.load(f)
    else:
        try:
            with open(cfg.model_weights_path, "rb") as f:
                params = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "No parameters file /app/params.pkl found. Please set load_from_hf to True."
            )
    return params


def create_fasta_from_sequences(
    l_seq: str, h_seq: str, output_file: str = "sequences.fasta"
) -> None:
    """
    Create a FASTA file from light and heavy chain sequences.

    Args:
        l_seq (str): Light chain sequence
        h_seq (str): Heavy chain sequence
        output_file (str, optional): Output FASTA file name. Defaults to "sequences.fasta".
    """
    l_seq = l_seq.strip()
    h_seq = h_seq.strip()

    if not l_seq or not h_seq:
        raise ValueError("Both light and heavy chain sequences must be provided")

    fasta_content = f">L_chain\n{l_seq}\n>H_chain\n{h_seq}\n"

    try:
        with open(output_file, "w") as f:
            f.write(fasta_content)
    except IOError as e:
        logging.error(f"Error writing FASTA file: {e}", exc_info=True)
