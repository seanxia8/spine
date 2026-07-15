"""Module with all the backbone components of UResNet+Attn.

Contains the following components:
  - `UResNetAttnEncoder`: Encoder component of UResNet+Attn
  - `UResNetAttnDecoder`: Decoder component of UResNet+Attn
  - `UResNetAttn`: Full encoder/decoder architecture of UResNet
"""

from typing import List
import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

import MinkowskiEngine as ME

from .act_norm import act_factory, norm_factory
from .blocks import ResNetBlock, AttentionBlock, ClusterAwareAttn
from .configuration import setup_cnn_configuration
from spine.utils.cluster.helpers import mask_cluster_id

__all__ = ['UResNetAttnEncoder', 'UResNetAttnDecoder', 'UResNetAttn']


class UResNetAttnEncoder(torch.nn.Module):
    """UResNet+Attention encoder.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg):
        """Initialize the encoder.

        Parameters
        ----------
        cfg : dict
            Encoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the input layer
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input, out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=self.dim,
            bias=self.allow_bias)

        # Initialize encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.num_planes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(
                    F, F, dimension=self.dim, activation=self.act_cfg,
                    normalization=self.norm_cfg, bias=self.allow_bias))
            m = torch.nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < (self.depth-1):
                m.append(norm_factory(self.norm_cfg, F))
                m.append(act_factory(self.act_cfg))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.num_planes[i],
                    out_channels=self.num_planes[i+1],
                    kernel_size=2, stride=2,
                    dimension=self.dim, bias=self.allow_bias))
            m = torch.nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = torch.nn.Sequential(*self.encoding_conv)
        self.encoding_block = torch.nn.Sequential(*self.encoding_block)
    def forward(self, x):
        """Pass a tensor through the encoder.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor
        attn_mask: input attention mask
        Returns
        -------
        encoder_tensors : List[ME.SparseTensor]
            List of intermediate tensors (taken between encoding block and
            convolution) from the encoder half
        final_tensor : ME.SparseTensor
            Feature tensor at deepest layer
        """
        x = self.input_layer(x)
        encoder_tensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoder_tensors.append(x)
            x = self.encoding_conv[i](x)

            if self.dropout:
                dropout_rate = min(0.4, 0.2+0.05*i)
                x = ME.MinkowskiDropout(dropout_rate)(x)
                #print(f"[Encoder] layer {i} dropout rate: {dropout_rate:.2f}")
        #attn = self.attention_block(x)
        result = {
            'encoder_tensors': encoder_tensors,
            #'attention_tensor': attn,
            #'gate_tensor': gate,
            'final_tensor': x
        }

        return result

class UResNetAttnDecoder(torch.nn.Module):
    """Vanilla UResNet Decoder.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg, enable_attention=True):
        """Initialize the decoder.

        Parameters
        ----------
        cfg : dict
            Decoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)
        #bottleneck_c = self.num_planes[-1]
        top_c = self.num_planes[0]
        ##self.bn_attn = AttentionBlock(in_channels=top_c, out_channels=top_c,
        #                              dimension=self.dim, embed_dim=self.embed_dim,
        #                              num_heads=self.num_heads, lower=self.lower, upper=self.upper)
        self.enable_attention = enable_attention
        use_hard_mask = getattr(self, "use_hard_mask", False)
        attn_gain     = getattr(self, "attn_gain", 0.5)
        analysis_mode = getattr(self, "analysis_mode", False)
        self.use_hard_mask = use_hard_mask
        self.checkpoint_attn = getattr(self, "checkpoint_attn", True)
        # Initialize decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(norm_factory(
                self.norm_cfg, self.num_planes[i+1]))
            m.append(act_factory(self.act_cfg))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.num_planes[i+1],
                out_channels=self.num_planes[i],
                kernel_size=2, stride=2, dimension=self.dim,
                bias = self.allow_bias))
            m = torch.nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.num_planes[i] * (2 if j == 0 else 1),
                                     self.num_planes[i], dimension=self.dim,
                                     activation=self.act_cfg,
                                     normalization=self.norm_cfg,
                                     bias=self.allow_bias))
            m = torch.nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = torch.nn.Sequential(*self.decoding_block)
        self.decoding_conv = torch.nn.Sequential(*self.decoding_conv)
        if self.enable_attention:
            self.bn_attn = ClusterAwareAttn(
                in_channels=top_c, dimension=self.dim,
                embed_dim=self.embed_dim, num_heads=self.num_heads,
                use_hard_mask=use_hard_mask, attn_gain=attn_gain,
                analysis_mode=analysis_mode)
        else:
            self.bn_attn = None
    '''    
    def _sparse_pc_mapping(self, g: ME.SparseTensor, tgt: ME.SparseTensor) -> ME.SparseTensor:
        """
        Non-learned NN upsample: copy each low-res parent value to all its
        high-res children using the coord manager's transpose kernel map.
        """
        cm  = g.coordinate_manager
        ik  = g.coordinate_map_key
        ok  = tgt.coordinate_map_key
        dev = g.F.device
        offsets = torch.empty((0, tgt.D), dtype=torch.int32)
        # Build the transpose kernel map low -> high (one stride-2 up)
        kmap = cm.get_kernel_map(
            ik, ok,
            [2] * tgt.D,   # kernel_size
            [2] * tgt.D,   # stride
            [1] * tgt.D,   # dilation
            ME.RegionType.HYPER_CUBE,
            offsets,
            True,          # is_transpose
            False          # is_pool
        )
        w_parent = torch.softmax(g.F, dim=1)
        outF = torch.zeros((tgt.F.shape[0], 1), device=dev, dtype=g.F.dtype)

        # kmap is {offset: tensor}. Tensor can be (2, nnz) or (nnz, 2) int32 on CPU.
        for offset_idx, ij in kmap.items():
            if ij.numel() == 0:
                continue
            ij = ij.view(2, -1) if ij.dim() != 2 else ij
            if ij.size(0) != 2:
                ij = ij.t()

            ii = ij[0].to(device=dev, dtype=torch.long)
            oo = ij[1].to(device=dev, dtype=torch.long)
            wk = w_parent[ii, offset_idx].unsqueeze(1)
            outF[oo] = wk


        return ME.SparseTensor(
            features=outF,
            coordinate_map_key=ok,
            coordinate_manager=cm,
        )
    '''
    def forward(self, x_bn: ME.SparseTensor, encoder_tensors, cluster_ids=None,
                node_confidence=None):
        """Pass a tensor through the decoder.

        Parameters
        ----------
        x_bn : ME.SparseTensor
            Output of the encoder
        encoder_tensors : List[ME.SparseTensor]
            List of tensors from each depth of the encoder
        cluster_ids : torch.Tensor, optional
            (N,) integer cluster assignment used to build the attention mask.
        node_confidence : torch.Tensor, optional
            (N,) per-voxel confidence in [0, 1] derived from edge probabilities.
            Passed to ``mask_cluster_id`` so that cross-cluster suppression is
            soft (proportional to confidence) rather than hard.  When ``None``
            the mask falls back to the original hard -inf / 0 behaviour.

        Returns
        -------
        List[ME.SparseTensor]
            List of feature tensors in decoding path at each spatial resolution
        """
        attn_tensors = []
        x = x_bn
        decoder_tensors = []

        for i, layer in enumerate(self.decoding_conv):
            skip = encoder_tensors[-i-2]
            x = layer(x)
            #g_scalar = self._sparse_mean_smooth(g_scalar, kernel_size=3, iters=2)
            #g_scalar = ME.MinkowskiSigmoid()(g_scalar)
            if i < (len(self.decoding_conv)-1) and self.dropout:
                dropout_rate = max(0.4-0.05*i,0.1)
                x = ME.MinkowskiDropout(dropout_rate)(x)
                #print(f"[Decoder] layer {i} dropout rate: {dropout_rate:.2f}")
            #g_scalar = self.offset_head(g_scalar)
            #g_scalar = self._sparse_pc_mapping(g_scalar, x)
            #g_scalar = ME.MinkowskiSigmoid()(logit)
            #attn_tensors.append(g_scalar)
            x = ME.cat(skip, x)
            x = self.decoding_block[i](x)

            #if i == len(self.decoding_conv) - 1:
            #    print(f"Decoder last layer input size: {x.F.shape[0]} points")

            # Apply cluster-aware attention at this decoder level
            if self.bn_attn is not None and cluster_ids is not None and i == len(self.decoding_conv) - 1:
                torch.cuda.empty_cache()
                if self.use_hard_mask:
                    # Hard {0, -inf} binary mask — no confidence weighting.
                    # Passing confidence=None forces create_*_mask to produce
                    # the binary mask, which is compatible with PyTorch's
                    # mem-efficient SDP backend (Flash SDP doesn't support
                    # arbitrary float additive masks but mem-efficient does).
                    attn_mask = mask_cluster_id(
                        x, cluster_ids,
                        confidence=None,
                        mode=self.attn_mode
                    )
                    if self.checkpoint_attn and self.training:
                        x, attn_weights = ckpt.checkpoint(
                            self.bn_attn, x, attn_mask, use_reentrant=False)
                    else:
                        x, attn_weights = self.bn_attn(x, attn_mask=attn_mask)
                    del attn_mask
                else:
                    # Soft cluster-direction path: no mask, Flash-compatible.
                    if self.checkpoint_attn and self.training:
                        x, attn_weights = ckpt.checkpoint(
                            self.bn_attn, x, None, cluster_ids, node_confidence,
                            use_reentrant=False)
                    else:
                        x, attn_weights = self.bn_attn(
                            x, cluster_ids=cluster_ids, confidence=node_confidence)
                attn_tensors.append([aw.detach() if aw is not None else None
                                     for aw in attn_weights])

                #torch.cuda.synchronize()
                #vram = torch.cuda.memory_allocated() / (1024 ** 2) - vram  # in MB
                #print(f"Attn score VRAM: {vram:.2f} MB")

            decoder_tensors.append(x)
            torch.cuda.empty_cache()

        return decoder_tensors, attn_tensors

class UResNetAttn(torch.nn.Module):
    """UResNet + Attention with access to intermediate feature planes.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg):
        """Initialize the UResNet backbone.

        Parameters
        ----------
        cfg : dict
            Decoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the encoder/decoder blocks of the UResNet model
        self.encoder = UResNetAttnEncoder(cfg)
        self.decoder = UResNetAttnDecoder(cfg)

    def forward(self, input_data, attn_pack=None):
        """Pass a tensor through the UResNet backbone.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor
        attn_pack: list of ME.SparseTensor
            Input sparse tensor for the attention
        Returns
        -------
        encoder_tensors : List[ME.SparseTensor]
            List of intermediate tensors (taken between encoding block and
            convolution) from the encoder half
        decoder_tensors : List[ME.SparseTensor]
            List of feature tensors in decoding path at each spatial resolution
        final_tensor : ME.SparseTensor
            Feature tensor at deepest layer
        """
        # Cast the input data to a sparse tensor
        coords = input_data[:, 0:self.dim + 1].int()
        features = input_data[:, self.dim + 1:]
        x = ME.SparseTensor(features, coordinates=coords)

        # Pass it through the encoder
        encoder_output = self.encoder(x)
        encoder_tensors = encoder_output['encoder_tensors']
        final_tensor = encoder_output['final_tensor']
        #encoder_attn_tensor = encoder_output['attention_tensor']

        # Pass it through the decoder
        decoder_tensors, decoder_attn_tensors = self.decoder(final_tensor, encoder_tensors, attn_pack)

        # Return
        res = {
            'encoder_tensors': encoder_tensors,
            'decoder_tensors': decoder_tensors,
            'final_tensor': final_tensor,
            'attn_tensors': decoder_attn_tensors
        }

        return res
