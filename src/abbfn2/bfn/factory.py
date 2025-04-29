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

from hydra.utils import instantiate
from omegaconf import DictConfig

from abbfn2.bfn.continuous import ContinuousBFN
from abbfn2.bfn.discrete import DiscreteBFN
from abbfn2.bfn.multimodal import MultimodalBFN

BFN = ContinuousBFN | DiscreteBFN | MultimodalBFN


def get_bfn(dm_cfg: DictConfig, output_network_cfg) -> BFN:
    """Get the BFN model.

    Args:
        dm_cfg (DictConfig): The data modes config.
        output_network_cfg (DictConfig): The output network config.

    Returns:
        BFN: The BFN model.
    """
    bfns = {
        dm: instantiate(cfg.bfn, output_network=None, _recursive_=False)
        for dm, cfg in dm_cfg.items()
    }
    bfn = MultimodalBFN(bfns, output_network_cfg)
    return bfn
