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

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from hydra.utils import instantiate
from jax import Array
from omegaconf import DictConfig

from abbfn2.bfn.types import (
    OutputNetworkPredictionMM,
    ThetaMM,
)


@dataclass
class BackboneConfig:
    """Configuration for the Backbone model.

    Attributes:
        embed_dim (int): Embedding dimension size.
        key_size (int): Key size for attention mechanisms.
        attention_heads (int): Number of attention heads.
        ffn_embed_dim (int): Feedforward embedding dimension size.
        num_layers (int): Number of layers in the model.
    """

    embed_dim: int = 1280
    key_size: int = 64
    attention_heads: int = 20
    ffn_embed_dim: int = 5120
    num_layers: int = 33


class BackboneRMSNorm(nn.RMSNorm):
    def __init__(
        self,
        epsilon: float = 1e-5,
        use_scale: bool = True,
        use_fast_variance: bool = True,
    ):
        super().__init__(
            epsilon=epsilon,
            use_scale=use_scale,
            use_fast_variance=use_fast_variance,
            dtype=jnp.float32,
        )

    def __call__(self, x):
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = super().__call__(x)
        return x.astype(x_dtype)


class RoPE(nn.Module):
    """Applies Rotary Position Embedding (RoPE).

    Args:
        max_wavelength (int): The base for the geometric progression used to
            compute the rotation angles.
    """

    max_wavelength: int

    def _apply_rope_1d(self, x: jax.Array, position_index: int):
        """Applies rotation to a vector at a specific position index.

        Args:
            x (jax.Array): Vector to rotate
            position_index (int): Position of the vector in the sequence.

        Returns:
            jax.Array: Rotated vector.
        """
        head_dim = x.shape[-1]
        fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        timescale = self.max_wavelength**fraction

        sinusoid_inp = position_index / timescale
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)

        first_half, second_half = jnp.split(x, 2)

        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return jnp.concatenate([first_part, second_part])

    def _apply_rope(self, x: jax.Array):
        """Vmap application of RoPE over all positions in the sequence.

        Args:
            x (jax.Array): Sequence of vectors

        Returns:
            jax.Array: Sequence with RoPE applied.
        """
        positions = jnp.arange(0, x.shape[0])
        return jax.vmap(self._apply_rope_1d)(x, positions)

    def __call__(self, x: jax.Array):
        """Apply RoPE to the input.

        Args:
            x (jax.Array): Input array

        Returns:
            jax.Array: Transformed input with RoPE applied.
        """
        head_batched = jax.vmap(
            self._apply_rope,
            in_axes=-2,
            out_axes=-2,
        )

        return head_batched(x.astype(jnp.float32)).astype(x.dtype)


class MergeNLastAndLinear(nn.Module):
    """Merge the last N dimensions of input before applying a Linear layer.

    Args:
        linear_out (int): Size of output dimension in the Linear layer.
        n_merge (int): Number of dimensions to merge.
    """

    linear_out: int
    n_merge: int

    def setup(self):
        """Set up the model components."""
        self._linear = nn.Dense(
            self.linear_out,
            kernel_init=nn.linear.default_kernel_init,
            use_bias=False,
            dtype=jnp.float32,
        )

    def __call__(self, x: jax.Array):
        """Forward pass.

        Args:
            x (jax.Array): Input array of shape (batch_size, sequence_length, head, head_dim)

        Returns:
            jax.Array: Output array after linear transformation of shape (batch_size, sequence_length, embed_dim).
        """
        # (B,T,H,D)
        x = x.reshape((*x.shape[: -self.n_merge], -1))
        y = self._linear(x)
        # (B,T,E)
        return y


class MultiHeadProjection(nn.Module):
    """Projection for key and query in Backbone models.

    Args:
        input_dim (int): Input dimension.
        num_head (int): Number of heads.
        head_size (int): Size of each head.
    """

    input_dim: int
    num_head: int
    head_size: int

    def setup(self):
        """Set up the model components."""
        self._linear = nn.Dense(
            self.num_head * self.head_size,
            use_bias=False,
            kernel_init=nn.linear.default_kernel_init,
            dtype=jnp.float32,
        )

    def __call__(self, x: jax.Array):
        """Forward pass.

        Args:
            x (jax.Array): Input array of shape (batch_size, sequence_length, embed_dim).

        Returns:
            jax.Array: Output array of shape (batch_size, sequence_length, num_head, head_size).
        """
        # (B,T,E)
        x = self._linear(x)
        x = x.reshape((*x.shape[:-1], self.num_head, self.head_size))
        # (B,T,H,D)
        return x


class Attention(nn.Module):
    """Attention block for the Backbone model.

    Args:
        project_query (Callable): Function to project input to the query space.
        project_key (Callable): Function to project input to the key space.
        project_value (Callable): Function to project input to the value space.
        compute_mha (Callable): Function to compute multi-head attention.
        project_output (Callable): Function to project the output of the multi-head attention.
        rotary_pos_emb (nn.Module): Instance of a Rotary Position Embedding layer.
    """

    project_query: Callable[[jax.Array], jax.Array]
    project_key: Callable[[jax.Array], jax.Array]
    project_value: Callable[[jax.Array], jax.Array]
    compute_mha: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    project_output: Callable[[jax.Array], jax.Array]
    rotary_pos_emb: nn.Module | None

    def setup(self):
        """Set up the model components."""
        self._rot_emb = None

        if self.rotary_pos_emb:
            self._rot_emb = self.rotary_pos_emb

    def __call__(self, x: jax.Array):
        """Forward pass.

        Args:
            x (jax.Array): Input array of shape (batch_size, sequence_length, embed_dim).

        Returns:
            jax.Array: Output array of shape (batch_size, sequence_length, embed_dim).
        """
        # (B,T,E)
        q = self.project_query(x)
        k = self.project_key(x)
        v = self.project_value(x)
        # (B,T,H,D)

        if self._rot_emb:
            q = self._rot_emb(q)
            k = self._rot_emb(k)

        mha_output = self.compute_mha(q, k, v)

        x = self.project_output(mha_output)
        # (B,T,E)
        return x


class MultiHeadAttention(nn.Module):
    """Configurable Multi-Head Attention module for Backbone model.

    Args:
        config (BackboneConfig): BackboneConfig dataclass containing model hyperparameters.
    """

    config: BackboneConfig

    def setup(self):
        """Set up the multi-head attention components."""
        key_size = self.config.key_size

        def _attention(q, k, v):
            """Compute multi-head attention using einsum."""
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)
            v = v.astype(jnp.float32)

            attn_logits = jnp.einsum("...thd,...Thd->...htT", q, k)
            sqrt_key_size = jnp.sqrt(key_size).astype(jnp.float32)
            attn_logits = attn_logits / sqrt_key_size
            attn_weights = jax.nn.softmax(attn_logits, axis=-1)
            ret = jnp.einsum("...htT,...Thd->...thd", attn_weights, v).astype(
                jnp.float32
            )
            return ret

        self.project_query = MultiHeadProjection(
            input_dim=self.config.embed_dim,
            num_head=self.config.attention_heads,
            head_size=key_size,
        )
        self.project_key = MultiHeadProjection(
            input_dim=self.config.embed_dim,
            num_head=self.config.attention_heads,
            head_size=key_size,
        )
        self.project_value = MultiHeadProjection(
            input_dim=self.config.embed_dim,
            num_head=self.config.attention_heads,
            head_size=key_size,
        )
        self.compute_mha = _attention

        self.project_output = MergeNLastAndLinear(
            linear_out=self.config.embed_dim,
            n_merge=2,
        )
        self.rotary_pos_emb = RoPE(max_wavelength=10000)

        self.attention_fn = Attention(
            project_query=self.project_query,
            project_key=self.project_key,
            project_value=self.project_value,
            compute_mha=self.compute_mha,
            project_output=self.project_output,
            rotary_pos_emb=self.rotary_pos_emb,
        )

    def __call__(self, x: jax.Array):
        """Forward pass for multi-head attention.

        Args:
            x (jax.Array): Input tensor.

        Returns:
            jax.Array: Output tensor after multi-head attention.
        """
        return self.attention_fn(x)


class Mlp(nn.Module):
    """MLP block for Backbone model's transformer block.

    Args:
        config (BackboneConfig): BackboneConfig dataclass containing model hyperparameters.
    """

    config: BackboneConfig

    def setup(self):
        """Set up the MLP layers."""
        ffn_embed_dim = int(2 / 3 * self.config.ffn_embed_dim) * 2

        self._fc1 = nn.Dense(
            ffn_embed_dim,
            use_bias=False,
            kernel_init=nn.linear.default_kernel_init,
            dtype=jnp.float32,
        )
        self._fc2 = nn.Dense(
            self.config.embed_dim,
            use_bias=False,
            kernel_init=nn.linear.default_kernel_init,
            dtype=jnp.float32,
        )
        self._activation = jax.nn.silu

    def __call__(self, x: jax.Array):
        """Forward pass through the MLP block.

        Args:
            x (jax.Array): Input tensor.

        Returns:
            jax.Array: Output tensor after MLP transformation.
        """
        fc1_act = self._fc1(x)
        x1, x2 = jnp.split(fc1_act, indices_or_sections=2, axis=-1)
        x = self._activation(x1) * x2
        x = self._fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Implementation of the transformer block for Backbone model.

    The block consists of a self-attention layer followed by a feedforward (MLP) layer.

    Attributes:
        self_attention (nn.Module): Attention block for self-attention.
        mlp (nn.Module): MLP block for feedforward layer.
        embed_dim (int): Embedding dimension.
    """

    self_attention: nn.Module
    mlp: nn.Module
    norm: nn.Module
    embed_dim: int

    def setup(self):
        """Set up the transformer block components."""
        self._layer_norm_1 = self.norm()
        self._layer_norm_2 = self.norm()

    def __call__(self, x: jax.Array):
        """Forward pass through the transformer block.

        Args:
            x (jax.Array): Input tensor.

        Returns:
            jax.Array: Output tensor after applying self-attention and MLP.
        """
        # Layer 1: MHA
        h = self._layer_norm_1(x)
        x = x + self.self_attention(h)

        # Layer 2: MLP
        residual = x
        x = self._layer_norm_2(x)

        x = self.mlp(x)
        x = x + residual

        return x


class BackboneBlock(nn.Module):
    """Specialization of a Transformer block for Backbone.

    Attributes:
        config (BackboneConfig): Configuration object for the Backbone model.
    """

    config: BackboneConfig

    def setup(self):
        """Set up the Backbone block components."""
        self.transformer_block = TransformerBlock(
            self_attention=MultiHeadAttention(self.config),
            mlp=Mlp(self.config),
            norm=BackboneRMSNorm,
            embed_dim=self.config.embed_dim,
        )

    def __call__(self, x: jax.Array):
        """Forward pass through the Backbone block.

        Args:
            x (jax.Array): Input tensor.

        Returns:
            jax.Array: Output tensor after applying the transformer block.
        """
        return self.transformer_block(x)


class BackboneScanBlock(BackboneBlock):
    """Specialization of a Transformer block for Backbone designed to work with scan loop.

    Attributes:
        config (BackboneConfig): Configuration object for the Backbone model.
    """

    def __call__(self, carry: jax.Array, _: None):
        """Forward pass through the Backbone block implemented for the nn.scan API.

        Args:
            carry jax.Array: input x
            _ (None): Not used.

        Returns:
            jax.Array: output after applying the transformer block.
        """
        x = self.transformer_block(carry)
        return x, None


class BackboneModel(nn.Module):
    """Backbone model with embeddings, layers, and language model head.

    Args:
        config (BackboneConfig): Configuration object for the Backbone model.
    """

    config: BackboneConfig

    def setup(self):
        """Set up the Backbone model components."""
        self._Backbone_block = nn.remat(
            target=BackboneScanBlock, policy=jax.checkpoint_policies.nothing_saveable
        )
        self._layers = nn.scan(
            self._Backbone_block,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            metadata_params={nn.PARTITION_NAME: "layers"},
            length=self.config.num_layers,
        )(self.config)

    def __call__(self, tokens):
        """Forward pass through the Backbone model.

        Args:
            tokens (jax.Array): Input token sequences.

        Returns:
            x: logits.
        """
        x, _ = self._layers(tokens, None)
        return x.astype(jnp.float32)


class TransformerBackbone(nn.Module):
    """AbBFN Transformer Backbone."""

    cfg: BackboneConfig

    @nn.compact
    def __call__(
        self,
        embeddings: dict[str, Array],
    ) -> tuple[Array, dict[str | Array]]:
        """Processes the input through the Transformer backbone.

        Args:
            embeddings (dict[str, Array]): Dictionary of input embeddings of shape (...var_shape..., dim).

        Returns:
            A dictionary containing the final embeddings.
        """
        # Define a canonical ordering of the data modes.
        data_modes = sorted(embeddings.keys())

        # Process the input embeddings into a (flat) format suitable for the Transformer.
        flat_embeddings, dm_var_shapes, dm_num_vars = [], [], []
        for dm in data_modes:
            *var_shape, dim = embeddings[dm].shape
            flat_dm_embeddings = embeddings[dm].reshape(-1, dim)

            flat_embeddings.append(flat_dm_embeddings)

            dm_var_shapes.append(tuple(var_shape))
            dm_num_vars.append(flat_dm_embeddings.shape[0])

        def split_and_reshape_fn(x):
            flat_embeddings = jnp.split(x, np.cumsum(dm_num_vars[:-1]))
            return [
                emb.reshape(var_shape + (-1,))
                for emb, var_shape in zip(flat_embeddings, dm_var_shapes)
            ]

        x = jnp.concatenate(flat_embeddings, axis=0)

        x = BackboneModel(self.cfg)(x)

        xs = split_and_reshape_fn(x)
        embeddings = {dm: xs[i] for i, dm in enumerate(data_modes)}

        return embeddings


class BFNMultimodalOutput(nn.Module):
    """Multimodal BFN."""

    bfn_cfgs: dict[str, DictConfig]
    network_cfg: DictConfig

    @nn.compact
    def __call__(
        self,
        theta: ThetaMM,
        t: float,
        beta: dict[str, Array],
    ) -> OutputNetworkPredictionMM:
        """Forward pass for Multimodal BFN."""
        data_modes = sorted(self.bfn_cfgs.keys())
        xs, skip_args = {}, {}
        dim = self.network_cfg.backbone.cfg.embed_dim

        for dm in data_modes:
            bfn_cfg = self.bfn_cfgs[dm]
            encoder = instantiate(bfn_cfg.encoder, output_dim=dim, name=f"encoder_{dm}")
            x, sa = encoder(
                theta[dm],
            )
            xs[dm] = x
            skip_args[dm] = sa

        backbone = instantiate(self.network_cfg.backbone, name="backbone")
        xs = backbone(xs)

        pred: OutputNetworkPredictionMM = {}
        for dm in data_modes:
            decoder = instantiate(self.bfn_cfgs[dm].decoder, name=f"decoder_{dm}")
            pred[dm] = decoder(xs[dm], skip_args[dm], t, beta[dm])
        return pred
