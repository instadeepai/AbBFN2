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
import time
import warnings

import hydra
import jax
import jax.numpy as jnp
import jax.random as random
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from abbfn2.bfn.factory import BFN, get_bfn
from abbfn2.data_mode_handler import save_samples
from abbfn2.utils.inference_utils import configure_output_dir, load_params

warnings.filterwarnings(
    "ignore",
    message=".*Explicitly requested dtype <class 'jax\\.numpy\\.float64'> requested in astype is not available.*",
    category=UserWarning,
)


@hydra.main(version_base="1.1", config_path="./configs", config_name="unconditional")
def main(full_config: DictConfig) -> None:
    """Main function.

    Args:
        full_config (DictConfig): The current configuration.
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

    # Prepare output directory.
    local_output_dir = configure_output_dir(cfg.output)

    # Inference function
    @jax.jit
    def batched_sample(params, key):
        key, sample_key = jax.random.split(key, 2)
        sample_fn = instantiate(cfg.sampling.sample_fn, bfn=bfn)
        sample_keys = jax.random.split(sample_key, cfg.sampling.num_samples_per_batch)
        sample_fn = jax.vmap(sample_fn, in_axes=(0, None))
        samples = sample_fn(sample_keys, params)
        return samples

    samples_raw = []
    logging.info("Beginning sampling")
    t = time.perf_counter()

    for _ in tqdm(range(cfg.sampling.num_batches)):
        key, key_sample = random.split(key, 2)
        samples = batched_sample(params, key_sample)
        samples_raw.append(jax.device_get(samples))

    def flatten_outputs(x):
        nb, bs, *dims = x.shape
        trg_shp = [
            nb * bs,
        ] + dims
        x = x.reshape(*trg_shp)
        return x

    samples_raw = jax.tree_util.tree_map(
        lambda *xs: flatten_outputs(jnp.stack(xs, axis=0)),
        *samples_raw,
    )

    sample_shapes = jax.tree_util.tree_map(lambda x: x.shape, samples_raw)
    num_samples = sample_shapes[next(iter(sample_shapes))][0]
    log_str = f"Generated {num_samples} samples in {time.perf_counter() - t:.2f}s:"
    logging.info(log_str)

    # Save samples and optionally raw info
    save_samples(samples_raw, dm_handlers, local_output_dir)

    if cfg.output.save_raw:
        logging.info(f"Saving raw samples to {local_output_dir}/raw.")
        for dm in dm_handlers.keys():
            dm_raw_dir = local_output_dir / "raw" / dm
            dm_raw_dir.mkdir(
                parents=True,
                exist_ok=cfg.output.overwrite_local_if_exists,
            )
            jnp.save(dm_raw_dir / "samples.npy", samples_raw[dm])


if __name__ == "__main__":
    main()
