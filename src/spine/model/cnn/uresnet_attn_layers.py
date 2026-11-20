"""UResNet backbone with cluster-conditioned attention in the decoder.

Contains the following components:
  - `UResNetAttnEncoder`: UResNet encoder accepting the attention config
  - `UResNetAttnDecoder`: UResNet decoder with attention after its last level
  - `UResNetAttn`: Full encoder/decoder architecture
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.checkpoint as ckpt

from spine.model import sparse
from spine.model.serial import SerializedAttention

from .attn_mask import mask_cluster_id
from .blocks import ClusterAwareAttn
from .uresnet_layers import UResNetDecoder, UResNetEncoder

__all__ = ["UResNetAttnEncoder", "UResNetAttnDecoder", "UResNetAttn"]

# Attention-only parameters and their defaults. `lower`/`upper` are accepted
# for backward compatibility with existing configurations but are unused.
ATTN_DEFAULTS: dict[str, Any] = {
    "embed_dim": 256,
    "num_heads": 8,
    "attn_gain": 0.75,
    "use_serial_attn": False,
    "attn_mode": "boundary",
    "checkpoint_attn": True,
    "use_hard_mask": False,
    "analysis_mode": False,
    "num_bits": 10,
    "attn_order": "hilbert",
    "attn_win_size": 256,
    "qkv_bias": False,
    "enable_flash": True,
    "lower": None,
    "upper": None,
}


def split_attn_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a backbone configuration into CNN and attention parameters.

    Returns
    -------
    dict
        Configuration accepted by :func:`setup_cnn_configuration`.
    dict
        Attention parameters, with defaults filled in.
    """
    cnn_cfg = {k: v for k, v in cfg.items() if k not in ATTN_DEFAULTS}
    attn_cfg = {k: cfg.get(k, v) for k, v in ATTN_DEFAULTS.items()}
    return cnn_cfg, attn_cfg


class UResNetAttnEncoder(UResNetEncoder):
    """UResNet encoder which ignores the attention-only parameters."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(split_attn_config(cfg)[0])


class UResNetAttnDecoder(UResNetDecoder):
    """UResNet decoder with cluster-conditioned attention after its last level.

    Attention is applied only when ``cluster_ids`` are provided. It uses
    either :class:`ClusterAwareAttn` (masked full attention per event) or
    :class:`SerializedAttention` (windowed attention along a space-filling
    curve), selected by ``use_serial_attn``.
    """

    def __init__(self, cfg: dict[str, Any], enable_attention: bool = True) -> None:
        cnn_cfg, attn_cfg = split_attn_config(cfg)
        super().__init__(cnn_cfg)
        for key, value in attn_cfg.items():
            setattr(self, key, value)

        self.enable_attention = enable_attention
        self.bn_attn = None
        if not enable_attention:
            return

        top_c = self.num_planes[0]
        if self.use_serial_attn:
            self.bn_attn = SerializedAttention(
                in_channels=top_c,
                dimension=self.dimension,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                window_size=self.attn_win_size,
                order=self.attn_order,
                num_bits=self.num_bits,
                attention_gain=self.attn_gain,
                qkv_bias=self.qkv_bias,
                analysis_mode=self.analysis_mode,
                enable_flash=self.enable_flash,
            )
        else:
            self.bn_attn = ClusterAwareAttn(
                in_channels=top_c,
                dimension=self.dimension,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                use_hard_mask=self.use_hard_mask,
                attn_gain=self.attn_gain,
                analysis_mode=self.analysis_mode,
            )

    def _attend(self, x, cluster_ids, node_confidence):
        """Apply the attention block, optionally with gradient checkpointing."""
        use_ckpt = self.checkpoint_attn and self.training

        def run(*args):
            if use_ckpt:
                return ckpt.checkpoint(self.bn_attn, *args, use_reentrant=False)
            return self.bn_attn(*args)

        if self.use_serial_attn:
            return run(x, cluster_ids)

        if not self.use_hard_mask:
            # Soft cluster-direction path: no mask, Flash-compatible
            return run(x, None, cluster_ids, node_confidence)

        # Hard {0, -inf} mask (no confidence weighting), compatible with the
        # memory-efficient SDP backend
        attn_mask = mask_cluster_id(
            x, cluster_ids, confidence=None, mode=self.attn_mode
        )
        return run(x, attn_mask)

    def forward(
        self,
        final: sparse.SparseTensor,
        encoder_tensors: list[sparse.SparseTensor],
        cluster_ids: torch.Tensor | None = None,
        node_confidence: torch.Tensor | None = None,
    ) -> tuple[list[sparse.SparseTensor], list[Any]]:
        """Decode the deepest representation, attending at the last level.

        Parameters
        ----------
        final : sparse.SparseTensor
            Deepest encoder representation.
        encoder_tensors : list of sparse.SparseTensor
            Encoder feature planes ordered from shallow to deep.
        cluster_ids : torch.Tensor, optional
            (N,) cluster assignment used to condition the attention.
        node_confidence : torch.Tensor, optional
            (N,) per-voxel clustering confidence in [0, 1].

        Returns
        -------
        list of sparse.SparseTensor
            Decoder feature planes ordered from deep to shallow.
        list
            Detached attention weights (one entry when attention ran).
        """
        attn_tensors = []
        decoder_tensors = []
        x = final
        last = len(self.decoding_conv) - 1
        for index, (upsample, block) in enumerate(
            zip(self.decoding_conv, self.decoding_block, strict=True)
        ):
            x = upsample(x)
            if self.dropout and index < last:
                x = self.decoding_dropout[index](x)
            x = sparse.cat(encoder_tensors[-index - 2], x)
            x = block(x)

            if self.bn_attn is not None and cluster_ids is not None and index == last:
                torch.cuda.empty_cache()
                x, attn_weights = self._attend(x, cluster_ids, node_confidence)
                if attn_weights is not None:
                    attn_tensors.append(
                        [aw.detach() if aw is not None else None for aw in attn_weights]
                    )
                else:
                    attn_tensors.append(None)

            decoder_tensors.append(x)
            torch.cuda.empty_cache()

        return decoder_tensors, attn_tensors


class UResNetAttn(torch.nn.Module):
    """UResNet + attention backbone with access to intermediate feature planes."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = UResNetAttnEncoder(cfg)
        self.decoder = UResNetAttnDecoder(cfg)
        self.dimension = self.encoder.dimension
        self.num_filters = self.encoder.num_filters

    def forward(self, input_data: torch.Tensor, attn_pack=None) -> dict[str, Any]:
        """Pass a batched voxel tensor through the backbone.

        Parameters
        ----------
        input_data : torch.Tensor
            (N, 1 + D + C) tensor of batch index, coordinates and features.
        attn_pack : torch.Tensor, optional
            (N,) cluster assignment forwarded to the decoder attention.
        """
        coords = input_data[:, 0 : self.dimension + 1].int()
        features = input_data[:, self.dimension + 1 :]
        x = sparse.SparseTensor(features, coordinates=coords)

        encoder_output = self.encoder(x)
        decoder_tensors, attn_tensors = self.decoder(
            encoder_output["final_tensor"],
            encoder_output["encoder_tensors"],
            attn_pack,
        )

        return {
            "encoder_tensors": encoder_output["encoder_tensors"],
            "decoder_tensors": decoder_tensors,
            "final_tensor": encoder_output["final_tensor"],
            "attn_tensors": attn_tensors,
        }
