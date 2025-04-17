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
