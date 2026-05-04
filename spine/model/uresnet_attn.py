"""UResNet segmentation model and its loss."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from spine.data import TensorBatch
from spine.utils.globals import BATCH_COL, COORD_COLS, VALUE_COL, GHOST_SHP
from spine.utils.logger import logger
from spine.utils.torch_local import cdist_fast

from .layer.factories import loss_fn_factory

from .layer.cnn.act_norm import act_factory, norm_factory
from .layer.cnn.uresnet_attn_layers import UResNetAttn

__all__ = ['UResNetAttnSegmentation', 'SegmentationLossAttn']


class UResNetAttnSegmentation(nn.Module):
    """UResNet+Attn for semantic segmentation.

    Typical configuration should look like:

    .. code-block:: yaml

        model:
          name: uresnet_attn
          modules:
            uresnet:
              # Your config goes here

    See :func:`setup_cnn_configuration` for available parameters for the
    backbone UResNet architecture.

    See configuration file(s) prefixed with `uresnet_` under the `config`
    directory for detailed examples of working configurations.
    """
    INPUT_SCHEMA = [
        ['sparse3d', (float,), (3, 1)]
    ]

    MODULES = ['uresnet_attn']

    def __init__(self, uresnet_attn, uresnet_attn_loss=None):
        """Initializes the standalone UResNet model.

        Parameters
        ----------
        uresnet_attn : dict
            Model configuration
        uresnet_attn_loss : dict, optional
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the model configuration
        self.process_model_config(**uresnet_attn)

        # Initialize the output layer
        self.output = [
            norm_factory(self.backbone.norm_cfg, self.num_filters),
            act_factory(self.backbone.act_cfg),
            ]
        self.output = nn.Sequential(*self.output)
        self.linear_segmentation = ME.MinkowskiLinear(
                self.num_filters, self.num_classes)

        # If needed, activate the ghost classification layer
        if self.ghost:
            logger.debug('Ghost Masking is enabled for UResNet Segmentation')
            self.linear_ghost = ME.MinkowskiLinear(self.num_filters, 2)

    def process_model_config(self, num_classes, attn_radius=None, ghost=False, **backbone):
        """Initialize the underlying UResNet model.

        Parameters
        ----------
        num_classes : int
            Number of classes to classify the voxels as
        attn_radius : float
            Attention mask radius
        ghost : bool, default False
            Whether to add a deghosting step in the classification model
        **backbone : dict
            UResNet backbone configuration
        """
        # Store the semantic segmentation configuration
        self.num_classes = num_classes
        self.ghost = ghost

        # Initialize the UResNetAttn backbone, store the relevant parameters
        self.backbone = UResNetAttn(backbone)
        self.num_filters = self.backbone.num_filters

    def forward(self, data, cluster_id):
        """Run a batch of data through the forward function.

        Parameters
        ----------
        data : TensorBatch
            (N, 1 + D + N_f) tensor of voxel/value pairs
            - N is the the total number of voxels in the image
            - 1 is the batch ID
            - D is the number of dimensions in the input image
            - N_f is the number of features per voxel

        Returns
        -------
        dict
            Dictionary of outputs
        """
        # Restrict the input to the requested number of features
        num_cols = 1 + self.backbone.dim + self.backbone.num_input
        input_tensor = data.tensor[:, :num_cols]

        # Pass the data through the UResNet backbone
        result_backbone = self.backbone(input_tensor, cluster_id)

        # Pass the output features through the output layer
        feats = result_backbone['decoder_tensors'][-1]
        feats = self.output(feats)
        seg   = self.linear_segmentation(feats)

        attn_tensor = result_backbone['attn_tensors'][-1]

        # Store the output as tensor batches
        segmentation = TensorBatch(seg.F, data.counts)
        attention_heatmap = TensorBatch(attn_tensor.F, data.counts)
        #print(f"At decoder output, the attention map has min: {attn_tensor.F.min()}, max: {attn_tensor.F.max()}.")

        batch_size = data.batch_size
        #final_tensor = TensorBatch(
        #        result_backbone['final_tensor'],
        #        batch_size=batch_size, is_sparse=True)
        #encoder_tensors = [TensorBatch(
        #    t, batch_size=batch_size,
        #    is_sparse=True) for t in result_backbone['encoder_tensors']]
        #decoder_tensors = [TensorBatch(
        #    t, batch_size=batch_size,
        #    is_sparse=True) for t in result_backbone['decoder_tensors']]
        #attn_tensors = [TensorBatch(
        #t, batch_size=batch_size,
        #is_sparse=True) for t in result_backbone['attn_tensors']]

        result = {
            'segmentation': segmentation,
            'attention_heatmap': attention_heatmap,
            #'final_tensor': final_tensor,
            #'encoder_tensors': encoder_tensors,
            #'decoder_tensors': decoder_tensors,
        }

        # If needed, pass the output features through the ghost linear layer
        if self.ghost:
            ghost = self.linear_ghost(feats)

            result['ghost'] = TensorBatch(ghost.F, data.counts)
            result['ghost_tensor'] = TensorBatch(
                    ghost, data.counts, is_sparse=True)

        return result


class SegmentationLossAttn(torch.nn.modules.loss._Loss):
    """Loss definition for semantic segmentation.

    For a regular flavor UResNet, it is a cross-entropy loss.
    For deghosting, it depends on a configuration parameter `ghost`:

    - If `ghost=True`, we first compute the cross-entropy loss on the ghost
      point classification (weighted on the fly with sample statistics). Then we
      compute a mask = all non-ghost points (based on true information in label)
      and within this mask, compute a cross-entropy loss for the rest of classes.

    - If `ghost=False`, we compute a N+1-classes cross-entropy loss, where N is
      the number of classes, not counting the ghost point class.

    See Also
    --------
    :class:`UResNetSegmentation`
    """
    INPUT_SCHEMA = [
        ['parse_sparse3d', (int,), (3, 1)]
    ]

    def __init__(self, uresnet_attn, uresnet_attn_loss):
        """
        Initializes the segmentation loss

        Parameters
        ----------
        uresnet_attn : dict
            Model configuration
        uresnet_attn_loss : dict
            Loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize what we need from the model configuration
        self.process_model_config(**uresnet_attn)

        # Initialize the loss configuration
        self.process_loss_config(**uresnet_attn_loss)

    def process_model_config(self, num_classes, ghost=False, **kwargs):
        """Process the parameters of the upstream model needed for in the loss.

        Parameters
        ----------
        num_classes : int
            Number of classes to classify the voxels as
        ghost : bool, default False
            Whether to add a deghosting step in the classification model
        **kwargs : dict, optional
            Leftover model configuration (no need in the loss)
        """
        # Store the semantic segmentation configuration
        self.num_classes = num_classes
        self.ghost = ghost

    def process_loss_config(self, loss='ce', ghost_label=-1, alpha=1.0,
                            beta=1.0, focal_gamma=1.0, focal_alpha=[1.0,1.0,1.0,1.0,1.0], balance_loss=False):
        """Process the loss function parameters.

        Parameters
        ----------
        loss : str, default 'ce'
            Loss function used for semantic segmentation
        ghost_label : int, default -1
            ID of ghost points. If specified (> -1), classify ghosts only
        alpha : float, default 1.0
            Classification loss prefactor
        beta : float, default 1.0
            Ghost mask loss prefactor
        balance_loss : bool, default False
            Whether to weight the loss to account for class imbalance
        focal_gamma : float, default None
            Gamma power for predicted label probability.

        """
        # Set the loss function

        try:
            self.loss_fn = loss_fn_factory(loss, reduction='none')
        except KeyError:
            raise ValueError(
                f"Unknown loss function `{loss}` provided. "
                f"Available options are: {list(loss_dict.keys())}")

        # Store the loss configuration
        self.ghost_label     = ghost_label
        self.alpha           = alpha
        self.beta            = beta
        self.balance_loss    = balance_loss
        self.focal_gamma     = focal_gamma
        self.focal_alpha     = focal_alpha

        # If a ghost label is provided, it cannot be in conjecture with
        # having a dedicated ghost masking layer
        assert not (self.ghost and self.ghost_label > -1), (
                "Cannot classify ghost exclusively (ghost_label) and "
                "have a dedicated ghost masking layer at the same time.")
    def forward(self, seg_label, segmentation, attention_heatmap, ghost=None, weights=None, **kwargs):
        """Computes the cross-entropy loss of the semantic segmentation
        predictions.

        Parameters
        ----------
        seg_label : TensorBatch
            (N, 1 + D + 1) Tensor of segmentation labels for the batch
        net_result : Dict of TensorBatch
            Diction of network outputs, must include `segmentation` and `attention_heatmap`
        ghost : TensorBatch, optional
            (N, 2) Tensor of ghost logits from the segmentation model
        weights : TensorBatch, optional
            (N) Tensor of weights for each pixel in the batch
        **kwargs : dict, optional
            Other outputs of the upstream model which are not relevant here

        Returns
        -------
        dict
            Dictionary of accuracies and losses
        """
        # Get the underlying tensor in each TensorBatch
        seg_label_t = seg_label.tensor
        segmentation_t = segmentation.tensor
        attn_heatmap_t = attention_heatmap.tensor
        ghost_t = ghost.tensor if ghost is not None else ghost
        weights_t = weights.tensor if weights is not None else weights

        # Make sure that the segmentation output and labels have the same length
        assert len(seg_label_t) == len(segmentation_t), (
                f"The `segmentation` output length ({len(segmentation_t)}) "
                f"and its labels ({len(seg_label_t)}) do not match.")
        assert len(segmentation_t) == len(attn_heatmap_t), (
                f"The `segmentation` output length ({len(segmentation_t)}) "
                f"and its attention gate weights ({len(attn_heatmap_t)}) do not match.")
        assert not self.ghost or len(seg_label_t) == len(ghost_t), (
                f"The `ghost` output length ({len(ghost_t)}) and "
                f"its labels ({len(seg_label_t)}) do not match.")
        assert not self.ghost or weights is None, (
                "Providing explicit weights is not compatible when peforming "
                "deghosting in tandem with semantic segmentation.")

        # Check that the labels have sensible values
        if self.ghost_label > -1:
            labels_t = (seg_label_t[:, VALUE_COL] == self.ghost_label).long()

        else:
            labels_t = seg_label_t[:, VALUE_COL].long()
            if torch.any(labels_t > self.num_classes):
                raise ValueError(
                        "The segmentation labels contain labels larger than "
                        "the number of logits output by the model.")

        # If there is a dedicated ghost layer, apply the mask first
        if self.ghost:
            # Count the number of voxels in each class
            ghost_labels_t = (labels_t == GHOST_SHP).long()
            ghost_loss, ghost_acc, ghost_acc_class = self.get_loss_accuracy(
                    ghost_t, ghost_labels_t)

            # Restrict the segmentation target to true non-ghosts
            nonghost = torch.nonzero(ghost_labels_t == 0).flatten()
            segmentation_t = segmentation_t[nonghost]
            labels_t = labels_t[nonghost]

        # Compute the loss/accuracy of the semantic segmentation step
        seg_loss, seg_acc, seg_acc_class, weights_t = self.get_loss_accuracy(
                segmentation_t, labels_t, weights_t)

        # Get the combined loss and accuracies
        result = {}
        if self.ghost:
            result.update({
                'loss': self.alpha * seg_loss + self.beta * ghost_loss,
                'accuracy': (seg_acc + ghost_acc)/2.,
                'seg_loss': seg_loss,
                'seg_accuracy': seg_acc,
                'ghost_loss': ghost_loss,
                'ghost_accuracy': ghost_acc,
                'ghost_accuracy_class_0': ghost_acc_class[0],
                'ghost_accuracy_class_1': ghost_acc_class[1]})

            for c in range(self.num_classes):
                result[f'seg_accuracy_class_{c}'] = seg_acc_class[c]

        else:
            result.update({
                'loss': seg_loss,
                'accuracy': seg_acc})

            for c in range(self.num_classes):
                result[f'accuracy_class_{c}'] = seg_acc_class[c]

        if weights_t is not None:
            result['weights'] = TensorBatch(weights_t, seg_label.counts)

        return result

    def get_loss_accuracy(self, logits, labels, weights=None):
        """Computes the loss, global and classwise accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            (N, N_c) Output logits from the network for each voxel
        labels : torch.Tensor
            (N) Target values for each voxel
        weights : torch.Tensor, optional
            (N) Tensor of weights for each pixel in the batch
        attn_wgts: torch.Tensor, optional
            (N) Tensor of attn gate weights for each pixel in the batch

        Returns
        -------
        torch.Tensor
            Cross-entropy loss value
        float
            Global accuracy
        np.ndarray
            (N_c) Vector of class-wise accuracy
        torch.Tensor
            (N) Updated set of weights for each pixel in the batch
        """
        # If there is no input, nothing to do
        num_classes = logits.shape[1]
        if not len(logits):
            return 0., 1., np.ones(num_classes, dtype=np.float32), weights

        # Count the number of voxels in each class
        counts = torch.empty(num_classes,
                dtype=torch.long, device=labels.device)
        for c in range(num_classes):
            counts[c] = torch.sum(labels == c).item()

        if weights is None:
            weights = torch.ones(labels.shape, device=labels.device)

        # If requested, create a set of weights based on class prevalance
        if self.balance_loss:
            class_weight = torch.ones(len(counts), dtype=logits.dtype, device=logits.device)
            m = counts > 0
            if m.any():
                cpos = counts[m].to(logits.dtype)
                # target cap
                cap = 3.0
                # effective exponent <= 0.5 to enforce max/min <= cap
                cmax = cpos.max()
                cmin = cpos.min()
                ratio = (cmax / (cmin + 1e-9)).clamp(min=1.0)
                alpha = torch.minimum(torch.tensor(0.5, dtype=logits.dtype, device=logits.device),
                                      (torch.log(torch.tensor(cap, dtype=logits.dtype, device=logits.device))
                                       / (torch.log(ratio + 1e-9) + 1e-12)))
                # inverse-frequency with adaptive exponent
                base = (len(labels) / num_classes)
                class_weight[m] = base / (cpos ** alpha)

            class_weights = class_weight[labels]
            weights = (weights * class_weights) if (weights is not None) else class_weights

        if isinstance(self.focal_alpha, (list, tuple, np.ndarray)):
            if len(self.focal_alpha) != num_classes:
                raise ValueError(f"focal alpha length {len(self.focal_alpha)} != num_classes {num_classes}")
            alpha_vec = torch.tensor(self.focal_alpha, dtype=logits.dtype, device=logits.device)
        else:
            # scalar
            alpha_vec = torch.full((num_classes,), float(self.focal_alpha),
                                   dtype=logits.dtype, device=logits.device)

        alpha_true = alpha_vec[labels]  # [N]

        log_p = torch.log_softmax(logits, dim=-1)  # [N, C]
        log_p_true = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)  # [N]
        p_true = log_p_true.exp().clamp(1e-6, 1-1e-6)

        # Focal term
        if self.focal_gamma is not None and self.focal_gamma > 0.0:
            focal = (1. - p_true).pow(self.focal_gamma)
            focal = focal / (focal.mean() + 1e-6)  # your normalization
        else:
            focal = torch.ones_like(p_true)

        #focal = focal / (focal.mean() + 1.e-6)
        per_voxel_loss = -log_p_true * focal * alpha_true

        # Compute the loss
        if hasattr(self.loss_fn, 'lambda_dice'):  # Check if it's a CE_DICE_Loss
            ce_losses, dice_loss = self.loss_fn(logits, labels)
            ce_losses = ce_losses * alpha_true * focal
            # Only weight the CE part with sample weights
            loss = (weights * ce_losses).sum() / weights.sum() + self.loss_fn.lambda_dice * dice_loss
        else:
            #loss = (weights * self.loss_fn(logits, labels)).sum() / weights.sum()
            loss = -log_p_true * focal
            if weights is not None:
                loss = (weights * per_voxel_loss).sum() / weights.sum()
            else:
                loss = per_voxel_loss.mean()

        # Compute the accuracies
        with torch.no_grad():
            # Per-class prediction accuracy
            preds = torch.argmax(logits, dim=-1)
            acc_class = np.ones(num_classes, dtype=np.float32)
            for c in range(num_classes):
                if counts[c] > 0:
                    mask = torch.nonzero(labels == c).flatten()
                    acc_class[c] = (preds[mask] == c).sum().item() / counts[c]

            # Global prediction accuracy
            acc = (preds == labels).sum().item() / torch.sum(counts).item()

        return loss, acc, acc_class, weights
