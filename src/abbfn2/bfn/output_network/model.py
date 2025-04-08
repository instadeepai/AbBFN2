from flax import linen as nn
from hydra.utils import instantiate
from jax import Array
from omegaconf import DictConfig

from abbfn2.bfn.output_network.backbone.backbone import TransformerBackbone
from abbfn2.bfn.types import (
    OutputNetworkPrediction,
    OutputNetworkPredictionMM,
    Theta,
    ThetaMM,
)


class BFNOutput(nn.Module):
    """Full BFN model. Equivalent to build_output_network_fn."""

    bfn_cfg: DictConfig
    network_cfg: DictConfig

    @nn.compact
    def __call__(
        self, theta: Theta, mask: dict[str, Array | None], t: float, beta: Array
    ) -> OutputNetworkPrediction:
        """Full BFN Forward Pass."""
        dim = self.network_cfg.backbone.embed_dim
        encoder = instantiate(self.bfn_cfg.encoder, output_dim=dim)

        x, skip_args = encoder(theta, t)

        x, _ = TransformerBackbone(self.network_cfg.backbone, name="backbone")(
            {"dm": x[None]},
            t,
            {"dm": mask},
        )
        x = x["dm"]

        decoder = instantiate(self.bfn_cfg.decoder)

        pred = decoder(x, skip_args, t, beta)

        return pred


class BFNMultimodalOutput(nn.Module):
    """Multimodal BFN. Equivalent to build_multimodal_output_network_fn."""

    bfn_cfgs: dict[str, DictConfig]
    network_cfg: DictConfig

    @nn.compact
    def __call__(
        self,
        theta: ThetaMM,
        mask: dict[str, Array | None],
        t: float,
        beta: dict[str, Array],
        pred_cond: OutputNetworkPredictionMM | None = None,
        t_cond: float | None = None,
    ) -> OutputNetworkPredictionMM:
        """Forward Pass Multimodal BFN."""
        data_modes = sorted(self.bfn_cfgs.keys())
        xs, skip_args = {}, {}
        dim = self.network_cfg.backbone.cfg.embed_dim

        for dm in data_modes:
            bfn_cfg = self.bfn_cfgs[dm]
            encoder = instantiate(bfn_cfg.encoder, output_dim=dim, name=f"encoder_{dm}")
            x, sa = encoder(
                theta[dm],
                t,
                mask[dm] if mask else None,
                pred_cond[dm] if pred_cond else None,
                t_cond,
            )
            xs[dm] = x
            skip_args[dm] = sa
            if "mask" in sa:
                mask[dm] = sa["mask"]

        backbone = instantiate(self.network_cfg.backbone, name="backbone")
        xs = backbone(xs, t, mask)

        pred: OutputNetworkPredictionMM = {}
        for dm in data_modes:
            decoder = instantiate(self.bfn_cfgs[dm].decoder, name=f"decoder_{dm}")
            pred[dm] = decoder(xs[dm], skip_args[dm], t, beta[dm], mask[dm] if mask else None)
        return pred
