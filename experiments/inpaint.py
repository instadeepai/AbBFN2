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
import math
import pickle
import time
import warnings

import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tabulate import tabulate
from tqdm import tqdm

from abbfn2.bfn import BFN, get_bfn
from abbfn2.data_mode_handler import save_samples
from abbfn2.data_mode_handler.sequence.sequence import SequenceDataModeHandler
from abbfn2.sample.functions import TwistedSDESampleFn
from abbfn2.sample.inpaint_masks import ConditionDataModeMaskFn, PredictDataModeMaskFn
from abbfn2.utils.inference_utils import (
    configure_imgt_position_overrides,
    configure_output_dir,
    flatten_and_crop,
    generate_random_mask_from_array_visible_pad,
    get_input_samples,
    load_params,
    pad_and_reshape,
    show_conditioning_settings,
)

warnings.filterwarnings(
    "ignore",
    message=".*Explicitly requested dtype <class 'jax\\.numpy\\.float64'> requested in astype is not available.*",
    category=UserWarning,
)


@hydra.main(version_base="1.1", config_path="./configs", config_name="inpaint.yaml")
def main(full_config: DictConfig) -> None:
    """Main function.

    Args:
        full_config (DictConfig): The current configuration.
    """
    cfg_run = full_config.run
    cfg = full_config.cfg

    key = random.PRNGKey(cfg.sampling.seed)

    # Build model.
    bfn: BFN = get_bfn(cfg_run.data_mode, cfg_run.output_network)
    key, bfn_key = random.split(key, 2)

    bfn.init(bfn_key)

    params = load_params(cfg.loading)

    # Initialise the data mode handlers.
    dm_handlers = {
        dm: instantiate(dm_cfg.handler) for dm, dm_cfg in cfg_run.data_mode.items()
    }

    # Prepare output directory.
    local_output_dir = configure_output_dir(cfg.output)

    # ======================== CREATE INPUT SAMPLES =======================

    with jax.default_device(jax.devices("cpu")[0]):
        key, input_key = random.split(key, 2)
        samples = get_input_samples(cfg.input, bfn, dm_handlers, input_key)
        num_samples = list(samples.values())[0].shape[0]

        mask_fn = instantiate(cfg.sampling.mask_fn)
        masks = mask_fn(samples, dm_handlers)
        for dm, handler in dm_handlers.items():
            if isinstance(handler, SequenceDataModeHandler):

                def get_sequence_mask(dm, mask_fn, padding_visible):
                    if isinstance(mask_fn, ConditionDataModeMaskFn):
                        # For conditioning, we want 1s (visible) if the DM is in data_modes
                        if dm in mask_fn.data_modes:
                            return np.ones(masks[dm].shape, dtype=int)
                        # Otherwise, we want 0s (hidden) unless padding is visible
                        return (
                            np.zeros(masks[dm].shape, dtype=int)
                            if not padding_visible
                            else None
                        )

                    if isinstance(mask_fn, PredictDataModeMaskFn):
                        # For prediction, we want 0s (hidden) if the DM is in data_modes
                        if dm in mask_fn.data_modes:
                            return (
                                np.zeros(masks[dm].shape, dtype=int)
                                if not padding_visible
                                else None
                            )
                        # Otherwise, we want 1s (visible)
                        return np.ones(masks[dm].shape, dtype=int)

                    return None

                new_mask = get_sequence_mask(dm, mask_fn, cfg.sampling.padding_visible)
                if new_mask is not None:
                    masks[dm] = new_mask

        # In case we want to add custom masks. These should be in a dictionary where
        # keys are DMs and values are custom masks of shape [n_samples, n_dims].
        if cfg.input.override_masks_file:
            override_mask_path = cfg.input.override_masks_file
            with open(override_mask_path, "rb") as rp:
                override_masks = pickle.load(rp)

            for dm, override_mask in override_masks.items():
                logging.info(f"Detected an override mask for dm: {dm}")
                new_mask = override_mask[
                    :num_samples, :
                ]  # the override mask has shape [n_samples, n_dims]
                logging.info(
                    f"{dm} mask shapes - old: {masks[dm].shape}, new: {new_mask.shape}"
                )
                masks[dm] = 1 - new_mask

        if "position_overrides" in cfg.input:
            parsed_overrides = {}
            for chain, overrides in cfg.input.position_overrides.items():
                for override_pos, override_res in overrides.items():
                    parsed_overrides[
                        (chain, (int(override_pos[:-1]), override_pos[-1]))
                    ] = override_res
            samples, masks = configure_imgt_position_overrides(
                masks, samples, parsed_overrides, dm_handlers
            )

        # Not compatible with custom masks being provided.
        if cfg.input.mask_weight_overrides:
            assert (
                not cfg.input.override_masks_file
            ), "Not compatible with custom masks being provided."
            for dm, mask_weight in cfg.input.mask_weight_overrides.items():
                logging.info(f"Detected a mask weight override: {dm}")
                if (
                    cfg.input.soft_inpaint_strategy == "random"
                ):  # If random, set to 1 or 0 based on expected conditioning freq.
                    masks[dm] = generate_random_mask_from_array_visible_pad(
                        samples[dm], frac_fill=mask_weight
                    )
                elif (
                    cfg.input.soft_inpaint_strategy == "soft"
                ):  # If soft, we directly pass the conditioning freq.
                    masks[dm] = np.where(
                        np.isin(samples[dm], [1, 4, 5]), 1, mask_weight
                    )

    initial_samples = samples

    show_conditioning_settings(num_samples, samples, masks)

    # Prepare inputs for batched sampling.
    num_batches = math.ceil(num_samples / cfg.sampling.num_samples_per_batch)
    num_samples_padded = num_batches * cfg.sampling.num_samples_per_batch

    logging.info(
        f"Padding from {num_samples} to {num_samples_padded} padded for {num_batches} batch(es) of {cfg.sampling.num_samples_per_batch} sample(s).",
    )
    samples = jax.tree_util.tree_map(
        lambda x: pad_and_reshape(x, num_samples_padded, num_batches), samples
    )
    masks = jax.tree_util.tree_map(
        lambda x: pad_and_reshape(x, num_samples_padded, num_batches), masks
    )

    inputs_info = {
        "num_batches": num_batches,
        "num_samples": num_samples,
        "samples": samples,
        "masks": masks,
    }

    # Prepare the batched sampling function.
    sample_fn = instantiate(cfg.sampling.inpaint_fn, bfn=bfn)

    # Inference function
    @jax.jit
    def batched_sample(params, key, x, mask):
        key, sample_key = jax.random.split(key, 2)
        sample_keys = jnp.array(
            jax.random.split(sample_key, cfg.sampling.num_samples_per_batch)
        )
        samples = jax.vmap(sample_fn, in_axes=(0, None, 0, 0))(
            sample_keys,
            params,
            x,
            mask,
        )
        return samples

    # Run sampling.
    def get_inputs(i):
        x = jax.tree_util.tree_map(lambda x: x[i], inputs_info["samples"])
        mask = jax.tree_util.tree_map(lambda x: x[i], inputs_info["masks"])
        return x, mask

    samples_raw = []
    logging.info("Beginning sampling")
    t = time.perf_counter()

    # =========================== MAIN INFERENCE LOOP =========================

    for i in tqdm(range(inputs_info["num_batches"])):
        x, mask = get_inputs(i)
        key, sample_key = jax.random.split(key, 2)
        samples = batched_sample(params, sample_key, x, mask)
        samples_raw.append(jax.device_get(samples))

    # ============================= POST-PROCESSING ===========================

    samples_raw = jax.tree_util.tree_map(
        lambda *xs: flatten_and_crop(jnp.stack(xs, axis=0)),
        *samples_raw,
    )
    masks = jax.tree_util.tree_map(flatten_and_crop, masks)

    log_header = f"Generated {inputs_info['num_samples']} samples in {time.perf_counter() - t:.2f}s."
    tab_data = [(dm, str(sample.shape)) for dm, sample in samples_raw.items()]
    tab_str = tabulate(
        tab_data,
        headers=["DataMode", "Sample"],
        tablefmt="rounded_outline",
    )
    log_str = log_header + "\n\t" + tab_str.replace("\n", "\n\t")

    if isinstance(sample_fn, TwistedSDESampleFn):
        # If we are using TwistedSDE inpainting; we need to handle the fact that each sample uses multiple particles.
        num_particles = sample_fn.num_particles
        num_samples_total = inputs_info["num_samples"] * num_particles
        log_header += f" Each sample used {num_particles} particles therefore {num_samples_total} particles generated."

        _prev_shape = dict(tab_data)
        samples_raw = jax.tree_util.tree_map(lambda x: x[:, 0, ...], samples_raw)
        tab_data = [
            (dm, _prev_shape[dm], str(sample.shape))
            for dm, sample in samples_raw.items()
        ]
        tab_str = tabulate(
            tab_data,
            headers=["DataMode", "Sample (SDE out)", "Sample"],
            tablefmt="rounded_outline",
        )
        log_str = log_header + "\n\t" + tab_str.replace("\n", "\n\t")

    logging.info(log_str)

    # Save samples and raw info.
    if cfg.sampling.force_conditioning_information:
        logging.info("Conditioning information will be enforced.")
        for dm, initial_sample in initial_samples.items():
            samples_raw[dm] = np.where(masks[dm] == 1, initial_sample, samples_raw[dm])

    save_samples(samples_raw, dm_handlers, local_output_dir)


if __name__ == "__main__":
    main()
