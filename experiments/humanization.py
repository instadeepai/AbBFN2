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
import os
import warnings
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from abbfn2.bfn import BFN, get_bfn
from abbfn2.data_mode_handler import save_samples
from abbfn2.utils.igblast import run_igblast_pipeline
from abbfn2.utils.inference_utils import (
    configure_output_dir,
    create_fasta_from_sequences,
    flatten_and_crop,
    get_input_samples,
    load_params,
    pad_and_reshape,
    show_conditioning_settings,
)
from abbfn2.utils.pre_humanisation import process_prehumanisation

warnings.filterwarnings(
    "ignore",
    message=".*Explicitly requested dtype <class 'jax\\.numpy\\.float64'> requested in astype is not available.*",
    category=UserWarning,
)


SEQ_DMS = [
    "h_fwr1_seq",
    "h_cdr1_seq",
    "h_fwr2_seq",
    "h_cdr2_seq",
    "h_fwr3_seq",
    "h_cdr3_seq",
    "h_fwr4_seq",
    "l_fwr1_seq",
    "l_cdr1_seq",
    "l_fwr2_seq",
    "l_cdr2_seq",
    "l_fwr3_seq",
    "l_cdr3_seq",
    "l_fwr4_seq",
]
FW_DMS = [
    "h_fwr1_seq",
    "h_fwr2_seq",
    "h_fwr3_seq",
    "h_fwr4_seq",
    "l_fwr1_seq",
    "l_fwr2_seq",
    "l_fwr3_seq",
    "l_fwr4_seq",
]
CDR_DMS = [
    "h_cdr1_seq",
    "h_cdr2_seq",
    "h_cdr3_seq",
    "l_cdr1_seq",
    "l_cdr2_seq",
    "l_cdr3_seq",
]


def process_input_overrides(dm_handlers, precursors, regions=None, num_devices=1):
    if regions is None:
        regions = SEQ_DMS

    # Collect sequences for each domain
    override_data = {
        dm: [seqs[dm.split("_")[0]][dm.split("_")[1]] for seqs in precursors.values()]
        for dm in regions
    }

    # Pad the data to make the total number divisible by num_devices
    num_samples = len(next(iter(override_data.values())))
    remainder = num_samples % num_devices
    if remainder != 0:
        pad_size = num_devices - remainder
        for dm in regions:
            last_row = override_data[dm][-1]
            override_data[dm].extend([last_row] * pad_size)

    # Generate override samples and masks
    override_samples = {
        dm: dm_handlers[dm].data_to_sample(override_data[dm]) for dm in regions
    }
    override_masks = {k: np.ones_like(v) for k, v in override_samples.items()}

    return override_samples, override_masks


def reweight_masks(masks, regions=None, weighting=1.0):
    if regions is None:
        regions = FW_DMS

    for dm in regions:
        masks[dm] = np.ones_like(masks[dm]) * weighting

    return masks


def generate_samples(
    samples, masks, num_batches, num_samples, cfg, bfn, key, params, num_samples_padded
):
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

    @jax.jit
    def batched_sample(params, key, x, mask):
        key, sample_key = jax.random.split(key, 2)
        sample_keys = jnp.array(
            jax.random.split(sample_key, cfg.sampling.num_samples_per_batch)
        )
        samples, preds = jax.vmap(sample_fn, in_axes=(0, None, 0, 0))(
            sample_keys,
            params,
            x,
            mask,
        )
        return samples, preds

    # Run sampling.
    def get_inputs(i):
        x = jax.tree_util.tree_map(lambda x: x[i], inputs_info["samples"])
        mask = jax.tree_util.tree_map(lambda x: x[i], inputs_info["masks"])
        return x, mask

    samples_raw = []
    preds_raw = []

    for j in range(inputs_info["num_batches"]):
        x, mask = get_inputs(j)
        key, sample_key = jax.random.split(key, 2)
        samples, preds = batched_sample(params, sample_key, x, mask)
        samples_raw.append(jax.device_get(samples))
        preds_raw.append(jax.device_get(preds))

    samples_raw = jax.tree_util.tree_map(
        lambda *xs: flatten_and_crop(jnp.stack(xs, axis=0)),
        *samples_raw,
    )

    preds_raw = jax.tree_util.tree_map(
        lambda *xs: flatten_and_crop(jnp.stack(xs, axis=0)),
        *preds_raw,
    )
    masks = jax.tree_util.tree_map(flatten_and_crop, masks)
    return samples_raw, preds_raw, masks, key, samples


def logistic_decay(t, n, max_val, min_val, steepness=6):
    return min_val + (max_val - min_val) * (1 / (1 + np.exp(steepness * (t / n - 0.5))))


def has_valid_vfams(cfg):
    """Check if valid vfams exist in config."""
    return (
        cfg.input.get("h_vfams", None) is not None
        or cfg.input.get("l_vfams", None) is not None
    )


@hydra.main(
    version_base="1.1", config_path="./configs", config_name="humanization.yaml"
)
def main(full_config: DictConfig) -> None:
    """Main function.

    Args:
        cfg (DictConfig): The current configuration.
    """
    cfg_run = full_config.run
    cfg = full_config.cfg

    key = random.PRNGKey(cfg.sampling.seed)

    bfn: BFN = get_bfn(cfg_run.data_mode, cfg_run.output_network)
    key, bfn_key = random.split(key, 2)

    bfn.init(bfn_key)

    params = load_params(cfg.loading)

    # Initialise the data mode handlers.
    dm_handlers = {
        dm: instantiate(dm_cfg.handler) for dm, dm_cfg in cfg_run.data_mode.items()
    }

    # Set up devices.
    devices = jax.local_devices()
    NUM_DEVICES = len(devices)
    logging.info(f"Found {NUM_DEVICES} local devices.")

    # Create output directories for intermediate results
    local_output_dir = os.path.abspath(configure_output_dir(cfg.output))

    # ================= FIXED HYPERPARAMETERS ==================
    SAMPLING_CFG = {
        "delta_decay_to": 0.5,
        "delta_decay_over": 5,
        "min_cond": 0.25,
        "hum_cond_logit_bounds": (0, 1),
    }

    OmegaConf.set_struct(cfg, False)

    cfg.input.num_input_samples = 1
    cfg.input.dm_overwrites = {"species": "human"}
    cfg.input.path = None

    cfg.sampling.inpaint_fn._target_ = "abbfn2.sample.functions.SDESampleFn"

    if "mask_fn" not in cfg.sampling:
        cfg.sampling.mask_fn = {}

    cfg.sampling.mask_fn._target_ = (
        "abbfn2.sample.inpaint_masks.ConditionDataModeMaskFn"
    )
    cfg.sampling.mask_fn.data_modes = ["species"]
    cfg.sampling.inpaint_fn.score_scale = {
        k: 16.0 for k in cfg.sampling.mask_fn.data_modes
    }
    cfg.sampling.num_samples_per_batch = cfg.input.num_input_samples

    cfg.enforce_cdr_sequence = True

    OmegaConf.set_struct(cfg, True)

    # ==================== PRE-HUMANIZATION ========================
    recycling_steps = cfg.sampling.recycling_steps
    l_seq = cfg.input.l_seq
    h_seq = cfg.input.h_seq

    # IgBLAST
    if not has_valid_vfams(cfg):
        fasta_file = "sequences.fasta"
        create_fasta_from_sequences(
            l_seq, h_seq, fasta_file
        )  # Dumps L and H string into a fasta format.
        vfams = run_igblast_pipeline(fasta_file)
        h_vfams = (
            vfams[1::2] if "h_vfams" not in cfg.input else cfg.input.h_vfams
        )  # Take odd indices (1, 3, 5, ...) for heavy chain
        l_vfams = (
            vfams[0::2] if "l_vfams" not in cfg.input else cfg.input.l_vfams
        )  # Take even indices (0, 2, 4, ...) for light chain
    else:
        h_vfams = cfg.input.h_vfams
        l_vfams = cfg.input.l_vfams
    assert len(h_vfams) == len(
        l_vfams
    ), "The number of vfams for the two chains are not the same"

    cfg.input.num_input_samples = len(l_vfams)

    # Pre-Humanization
    prehumanised_data, non_prehumanised_data = process_prehumanisation(
        input_heavy_seqs=[h_seq],
        input_light_seqs=[l_seq],
        hv_families=h_vfams,
        lv_families=l_vfams,
    )

    # Prepare input samples and masks.
    with jax.default_device(jax.devices("cpu")[0]):
        key, input_key = random.split(key, 2)
        samples = get_input_samples(cfg.input, bfn, dm_handlers, input_key)
        num_samples = list(samples.values())[0].shape[0]

        mask_fn = instantiate(cfg.sampling.mask_fn)
        masks = mask_fn(samples, dm_handlers)
        # Need to manually set the seq masks.
        for dm in SEQ_DMS:
            if dm not in cfg.sampling.mask_fn.data_modes:
                masks[dm] = np.zeros(masks[dm].shape, dtype=int)

    # Define and apply the sequence overrides
    override_samples, override_masks = process_input_overrides(
        dm_handlers, prehumanised_data, num_devices=NUM_DEVICES
    )

    override_samples["hv_family"] = dm_handlers["hv_family"].data_to_sample(h_vfams)
    override_samples["lv_family"] = dm_handlers["lv_family"].data_to_sample(l_vfams)
    override_masks["hv_family"] = np.ones_like(masks["hv_family"])
    override_masks["lv_family"] = np.ones_like(masks["lv_family"])

    for dm in SEQ_DMS + ["hv_family", "lv_family"]:
        samples[dm] = override_samples[dm]
        masks[dm] = override_masks[dm]

    show_conditioning_settings(num_samples, samples, masks)

    num_batches = math.ceil(num_samples / cfg.sampling.num_samples_per_batch)
    num_samples_padded = num_batches * cfg.sampling.num_samples_per_batch

    # Saved to enforce CDR sequence
    initial_samples = samples.copy()

    precursor_data = {}
    prec_samples, _ = process_input_overrides(
        dm_handlers, non_prehumanised_data, num_devices=NUM_DEVICES
    )
    for dm in SEQ_DMS:
        precursor_data[dm] = dm_handlers[dm].sample_to_data(prec_samples[dm])

    # Save mutations made in pre-humanization for conditional masking
    prehum_positions = {}
    for dm in FW_DMS:
        prehum_positions[dm] = override_samples[dm] != prec_samples[dm]

    humanness_vals = []
    nbr_mutations = []

    # ======== RUNNING THROUGH THE MODEL ONCE TO GET INITIAL HUMANNESS LEVEL =========

    # Create the initial override masks
    override_masks = {dm: np.ones_like(masks[dm]) for dm in FW_DMS}

    # Get the initial predictions and calculate the sequence-level conditioning scale
    _, baseline_preds_raw, masks, key, _ = generate_samples(
        samples,
        masks,
        num_batches,
        num_samples,
        cfg,
        bfn,
        key,
        params,
        num_samples_padded,
    )
    humanness = (
        baseline_preds_raw["species"]
        .to_distribution()
        .probs[:, :, 0]
        .reshape(samples["species"].shape)
    )
    hum_cond_vals = np.clip(
        np.interp(humanness, SAMPLING_CFG["hum_cond_logit_bounds"], (0, 1)),
        SAMPLING_CFG["min_cond"],
        1,
    )
    humanness_vals.append(np.array(humanness).tolist()[0][0])
    logging.info(f"initial humanness: {humanness_vals[-1]}")

    # Initialise the ages tree to keep track of when positions were changed - start ages off high
    ages = jax.tree_util.tree_map(lambda x: np.full(x.shape, 10), samples)
    ages = {k: v for k, v in ages.items() if k in FW_DMS}

    # Identify changed positions
    changed_positions = jax.tree_util.tree_map(
        lambda x1, x2: x1 != x2,
        {k: v for k, v in samples.items() if k in FW_DMS},
        {k: v for k, v in prec_samples.items() if k in FW_DMS},
    )
    nbr_mutations.append(0)

    # Increment the ages tree if a position has been changed
    ages = jax.tree_util.tree_map(lambda x: x + 1, ages)

    # Reset ages where changes have been made
    def update_age(x1, x2):
        x1[x2] = 0
        return x1

    ages = jax.tree_util.tree_map(update_age, ages, changed_positions)

    # Create the conditioning factors based on ages
    age_cond = jax.tree_util.tree_map(
        lambda x: logistic_decay(
            x, SAMPLING_CFG["delta_decay_over"], 1, SAMPLING_CFG["delta_decay_to"]
        ),
        ages,
    )

    def threshold_age_cond(x1, x2):
        x1[x2 > SAMPLING_CFG["delta_decay_over"]] = SAMPLING_CFG["min_cond"]
        np.clip(x1, SAMPLING_CFG["min_cond"], 1.0)
        return x1

    age_cond = jax.tree_util.tree_map(threshold_age_cond, age_cond, ages)

    # Create the conditioning factors based on humanness
    hum_cond = {
        k: np.clip(v * hum_cond_vals, SAMPLING_CFG["min_cond"], 1.0)
        for k, v in override_masks.items()
    }
    cond_masks = jax.tree_util.tree_map(
        lambda x1, x2: np.maximum(x1, x2), age_cond, hum_cond
    )
    cond_masks = jax.tree_util.tree_map(
        lambda x1, x2: np.maximum(x1, x2), cond_masks, prehum_positions
    )

    # ============================= MAIN INFERENCE LOOP ===========================

    logging.info("Beginning sampling")
    for i in tqdm(range(recycling_steps)):
        step_dir = local_output_dir + f"/step_{i}"
        os.makedirs(step_dir, exist_ok=True)

        # Set the species label at the beginning of each recycling step
        samples["species"] = initial_samples["species"]
        for dm in ["hv_family", "lv_family"]:
            samples[dm] = initial_samples[dm]
            masks[dm] = np.ones_like(masks[dm]) * 0.5

        # Set the conditioning masks according to the changes made in the previous iteration
        for dm in FW_DMS:
            masks[dm] = cond_masks[dm]

        prev_samples = samples
        samples_raw, preds_raw, masks, key, samples = generate_samples(
            prev_samples,
            masks,
            num_batches,
            num_samples,
            cfg,
            bfn,
            key,
            params,
            num_samples_padded,
        )

        # Create the override masks
        override_masks = {dm: np.ones_like(masks[dm]) for dm in FW_DMS}
        humanness = (
            preds_raw["species"]
            .to_distribution()
            .probs[:, :, 0]
            .reshape(samples["species"].shape)
        )
        hum_cond_vals = np.clip(
            np.interp(humanness, SAMPLING_CFG["hum_cond_logit_bounds"], (0, 1)),
            SAMPLING_CFG["min_cond"],
            1,
        )
        humanness_vals.append(np.array(humanness).tolist()[0][0])
        logging.info(f"humanness: {humanness_vals[-1]}")

        # Identify changed positions
        changed_positions = jax.tree_util.tree_map(
            lambda x1, x2: x1 != x2,
            {k: v for k, v in samples.items() if k in FW_DMS},
            {k: v for k, v in prev_samples.items() if k in FW_DMS},
        )
        nbr_mutations.append(int(sum(cp.sum() for cp in changed_positions.values())))

        # Increment the ages tree if a position has been changed
        ages = jax.tree_util.tree_map(lambda x: x + 1, ages)

        # Reset ages where changes have been made
        ages = jax.tree_util.tree_map(update_age, ages, changed_positions)

        # Create the conditioning factors based on ages
        age_cond = jax.tree_util.tree_map(
            lambda x: logistic_decay(
                x, SAMPLING_CFG["delta_decay_over"], 1, SAMPLING_CFG["delta_decay_to"]
            ),
            ages,
        )
        age_cond = jax.tree_util.tree_map(threshold_age_cond, age_cond, ages)

        # Create the conditioning factors based on humanness
        hum_cond = {
            k: np.clip(v * hum_cond_vals, SAMPLING_CFG["min_cond"], 1.0)
            for k, v in override_masks.items()
        }
        cond_masks = jax.tree_util.tree_map(
            lambda x1, x2: np.maximum(x1, x2), age_cond, hum_cond
        )
        cond_masks = jax.tree_util.tree_map(
            lambda x1, x2: np.maximum(x1, x2), cond_masks, prehum_positions
        )

        if cfg.sampling.enforce_cdr_sequence:
            for dm in CDR_DMS:
                samples_raw[dm] = initial_samples[dm]

        save_samples(samples_raw, dm_handlers, Path(step_dir))

        # Early stopping
        if humanness_vals[-1] > 0.95:
            logging.info("Humanness level reached 95% - stopping")
            break

    logging.info(f"Hummaness values on each recycling step: {humanness_vals}")
    logging.info(f"Number of mutations on each recycling step: {nbr_mutations}")


if __name__ == "__main__":
    main()
