from typing import Union

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time as time
from typing import Optional


try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .base import encode, decode
from .utils import offset2bincount, bincount2offset


__all__ = [
    "SerializedAttention",
]

class SerializedAttention(ME.MinkowskiNetwork):

    def __init__(
        self, 
        in_channels, 
        embed_dim, 
        num_heads, 
        scale=None,
        qkv_bias=False,        
        window_size=-1,
        dimension=3,
        order="hilbert",
        num_bits=8,
        attention_gain=0.5,
        analysis_mode=False,
        enable_flash=True,
        ):


        super().__init__(dimension)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.analysis_mode = analysis_mode
        self.num_bits = num_bits
        self.num_dims = dimension
        self.scale = scale if scale is not None else (embed_dim//num_heads) ** -0.5
        self.order = order
        self.qkv_bias = qkv_bias
        self.qkv_proj = nn.Linear(in_channels, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, in_channels, bias=False)

        self.gate = nn.Parameter(torch.tensor(attention_gain, requires_grad=True))
        self.enable_flash = enable_flash and (flash_attn is not None)

        if not self.enable_flash:
            self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def cluster_hilbert_order(self, coords:torch.Tensor, cluster_ids:torch.Tensor,
                              batch_ids:torch.Tensor, num_bits:int=8):
        """
        coords:      (N, 3) integer voxel coords
        cluster_ids: (N,)   integer, -1 = noise
        batch_ids:   (N,)   integer batch index (coords[:, 0])
        returns:     (N,)   argsort indices — batch → cluster → Hilbert
        """
        coords_shifted = coords - coords.min(dim=0).values
        coords_shifted = coords_shifted.clamp(0, 2**num_bits - 1)

        h = encode(coords_shifted, depth=num_bits, order=self.order)  # (N,) int64

        cid = cluster_ids.clone()
        cid[cid < 0] = int(cid.max()) + 1

        offset_h = 2 ** (self.num_dims * self.num_bits)   # 2^24 for num_bits=8, num_dims=3
        n_cid    = int(cid.max()) + 2
        sort_key = batch_ids.long() * (n_cid * offset_h) + cid.long() * offset_h + h

        return sort_key.argsort()

    @torch.no_grad()
    def get_padding_and_inverse(self, coords):
        batch_indices = coords[:, 0]
        bincount = torch.bincount(batch_indices.long())
        if self.window_size == -1:
            offset = bincount2offset(bincount)
            cu_seqlens = torch.cat([torch.tensor([0], device=offset.device, dtype=torch.long), offset]).int()
            return None, None, cu_seqlens

        # Round up every batch element to the next multiple of window_size.
        # The old code only padded when bincount[i] > window_size, leaving
        # smaller counts unaligned and making .view(n_win, K) fail.
        bincount_pad = (
            torch.div(bincount + self.window_size - 1, self.window_size, rounding_mode="trunc")
            * self.window_size
        )
        _offset = nn.functional.pad(bincount.cumsum(dim=0), (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=bincount_pad.device)
        unpad = torch.arange(_offset[-1], device=bincount_pad.device)
        cu_seqlens = []
        for i in range(len(bincount)):
            unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
            remainder = int(bincount[i] % self.window_size)
            if remainder != 0:
                n_pad     = self.window_size - remainder
                pad_start = int(_offset_pad[i + 1]) - n_pad
                if bincount[i] >= self.window_size:
                    # multi-window: copy from the same position in the previous window
                    pad[pad_start : _offset_pad[i + 1]] = pad[
                        pad_start - self.window_size : int(_offset_pad[i + 1]) - self.window_size
                    ]
                else:
                    # single incomplete window: repeat the first real token index
                    pad[pad_start : _offset_pad[i + 1]] = int(_offset_pad[i])
            pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=self.window_size,
                    dtype=torch.int32,
                    device=bincount.device,
                )
            )
        cu_seqlens = nn.functional.pad(
            torch.cat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )
        return pad, unpad, cu_seqlens

    def forward(self, x:ME.SparseTensor, cluster_ids:torch.Tensor=None, **kwargs):

        """Forward pass.
        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor.
        cluster_ids : torch.Tensor
            Cluster IDs for each voxel.
        """

        feats = x.F
        coords = x.C

        delta = torch.zeros_like(feats)

        if cluster_ids is not None:
            serial_order = self.cluster_hilbert_order(
                coords[:, 1:], cluster_ids, coords[:, 0], self.num_bits
            )
        else:
            return x

        H = self.num_heads
        E = self.embed_dim      # attention space (qkv_proj output / H = head_dim)
        K = self.window_size

        if self.enable_flash:
            # ── Path 1: flash_attn_varlen — one sequence per (batch, cluster) ──────
            # No windowing or padding: varlen handles variable-length sequences natively
            # and stores only O(N) log-sum-exp statistics for the backward, avoiding the
            # O(n_win·H·K²) attention-weight matrix that math SDP materialises.
            from flash_attn import flash_attn_varlen_qkvpacked_func

            N = feats.shape[0]
            feat_sorted = feats[serial_order]          # (N, C)
            qkv = self.qkv_proj(feat_sorted)           # (N, 3*E)

            # serial_order sorts batch-primary → cluster-secondary, so same-(batch,cluster)
            # voxels are contiguous; unique_consecutive gives exact per-cluster boundaries.
            batch_sorted = coords[serial_order, 0].long()
            cid_sorted   = cluster_ids[serial_order].clone()
            cid_sorted[cid_sorted < 0] = int(cid_sorted.max()) + 1
            n_cid_val    = int(cid_sorted.max()) + 1
            compound     = batch_sorted * n_cid_val + cid_sorted
            _, per_group_counts = torch.unique_consecutive(compound, return_counts=True)
            cu_seqlens_varlen = F.pad(
                per_group_counts.to(torch.int32).cumsum(0), (1, 0), value=0
            )
            max_seqlen_varlen = int(per_group_counts.max())

            # flash_attn requires fp16/bf16.
            # Cast before the flash call and delete the float32 source immediately so
            # the 3*E-wide float32 buffer (614 MB at N=200K, E=256) doesn't coexist
            # with the bf16 copy during the flash forward.  qkv's VALUE is not needed
            # for backward: flash saves qkv_packed (bf16), and qkv_proj's AddmmBackward
            # saves feat_sorted (the input), not the output.
            orig_dtype = qkv.dtype
            qkv_packed = qkv.view(N, 3, H, E // H)
            if orig_dtype not in (torch.float16, torch.bfloat16):
                qkv_packed = qkv_packed.to(torch.bfloat16)
                del qkv  # storage ref-count → 0; freed before flash allocates workspace

            out = flash_attn_varlen_qkvpacked_func(
                qkv_packed,
                cu_seqlens=cu_seqlens_varlen,
                max_seqlen=max_seqlen_varlen,
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )                                          # (N, H, E//H)
            flash_attn_qk = out.reshape(N, E).to(orig_dtype)

        else:
            # ── Path 2: windowed SDP with explicit cluster-boundary float mask ──────
            # Falls through to Path 3 (MHA) on exception (e.g. when math SDP is selected
            # and the float mask causes a dispatch error on this PyTorch build).
            pad, unpad, _ = self.get_padding_and_inverse(coords[serial_order])

            feat_sorted = feats[serial_order][pad]
            qkv = self.qkv_proj(feat_sorted)          # (N_pad, 3*E)
            n_win = qkv.shape[0] // K

            window_cid = cluster_ids[serial_order][pad].view(n_win, K)
            same       = (window_cid.unsqueeze(2) == window_cid.unsqueeze(1))
            attn_mask  = qkv.new_zeros(n_win, 1, K, K)
            attn_mask.masked_fill_(~same.unsqueeze(1), float('-inf'))

            q, k, v = qkv.view(n_win, K, 3, H, E // H).permute(2, 0, 3, 1, 4).unbind(0)

            try:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    scale=self.scale,
                )                                                        # (n_win, H, K, E//H)
                flash_attn_qk = out.permute(0, 2, 1, 3).reshape(-1, E) # (N_pad, E)
                flash_attn_qk = flash_attn_qk.to(qkv.dtype)
            except Exception:
                # ── Path 3: MHA fallback ──────────────────────────────────────────
                attn_mask_mha = (
                    attn_mask.expand(-1, H, -1, -1)
                    .reshape(n_win * H, K, K)
                )
                q_mha = q.permute(0, 2, 1, 3).reshape(n_win, K, E)
                k_mha = k.permute(0, 2, 1, 3).reshape(n_win, K, E)
                v_mha = v.permute(0, 2, 1, 3).reshape(n_win, K, E)
                flash_attn_qk = self.mha(
                    q_mha, k_mha, v_mha,
                    attn_mask=attn_mask_mha,
                    need_weights=self.analysis_mode,
                    average_attn_weights=False,
                )[0].to(q_mha.dtype)
                flash_attn_qk = flash_attn_qk.reshape(-1, E)

            if pad is not None:
                flash_attn_qk = flash_attn_qk[unpad]

        attn_out = self.out_proj(flash_attn_qk)
        delta[serial_order] = attn_out                
        out_feats = self.gate * delta + feats
        
        out_x = ME.SparseTensor(
            features=out_feats,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )
        if self.analysis_mode:
            return out_x, delta
        
        return out_x, None
    