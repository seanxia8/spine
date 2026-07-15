"""Attention head superposition analysis for ClusterAwareAttn.

This module answers the question: *which heads are responsible for which
semantic classes?*

Attention heads operate in feature space, so there is no direct mapping
"head h → class c".  Three complementary strategies are provided, ordered
from cheap (weight-only) to expensive (data-dependent):

Weight geometry (no data required)
    `HeadSuperpositionAnalyzer.weight_geometry`

    For each head h, the effective operator that maps head-h outputs to
    class-c logits is  ``W_seg[c,:] @ W_O[:,h*d:(h+1)*d]``, a vector of
    length ``head_dim``.  Its L2 norm tells you the *maximum possible*
    contribution of head h to class c.  A low norm means the model weights
    structurally prevent head h from influencing class c.

Attention pattern analysis (requires one forward pass with labels)
    `HeadSuperpositionAnalyzer.attention_class_affinity`

    Given per-head attention weights ``A_h[i,j]`` and ground-truth labels,
    compute the *class affinity matrix*  ``M_h[c1,c2] = mean A_h[i,j]``
    where ``class(i)=c1, class(j)=c2``.  The diagonal shows within-class
    concentration; a large diagonal means head h is "monosemantic".

Feature separability  (requires forward pass in analysis_mode=True)
    `HeadSuperpositionAnalyzer.feature_separability`

    After enabling ``model.decoder.bn_attn.analysis_mode = True``, each
    forward pass stores per-head V-weighted outputs ``(N, head_dim)`` in
    ``_last_head_outputs``.  We compute the Fisher LDA separability ratio
    for each (head, class) pair: a high ratio means class c is well
    separated in head h's feature subspace.  Multiple high-ratio classes
    on the same head → superposition.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HeadSuperpositionAnalyzer"]

# Labels used in SPINE (adjust if your project uses different IDs)
_DEFAULT_CLASS_NAMES = {
    0: "shower",
    1: "delta",
    2: "michel",
    3: "lowE",
    4: "track",
}


class HeadSuperpositionAnalyzer:
    """Analyse per-head semantic specialisation in ``ClusterAwareAttn``.

    Parameters
    ----------
    attn_module : ClusterAwareAttn
        The instantiated attention module from the model.
    seg_linear : ME.MinkowskiLinear
        The final segmentation linear layer (``model.linear_segmentation``).
    num_classes : int
        Number of semantic classes.
    class_names : dict, optional
        Mapping from class index to human-readable name.
    device : str or torch.device, optional
        Device to run computations on. Defaults to CPU.
    """

    def __init__(
        self,
        attn_module,
        seg_linear,
        num_classes: int = 5,
        class_names: Optional[Dict[int, str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.attn = attn_module
        self.seg_linear = seg_linear
        self.num_classes = num_classes
        self.class_names = class_names or _DEFAULT_CLASS_NAMES
        self.device = device or torch.device("cpu")

        self.num_heads: int = attn_module.num_heads
        self.head_dim: int = attn_module.head_dim
        self.embed_dim: int = attn_module.embed_dim

        # Accumulated data for affinity / separability analyses
        # Each element: list-of-batch tensors from one forward call.
        self._attn_weights_buffer: List[List[torch.Tensor]] = []   # (num_heads, N_b, N_b)
        self._head_outputs_buffer: List[List[torch.Tensor]] = []   # (N_b, num_heads, head_dim)
        self._labels_buffer: List[torch.Tensor] = []               # (N,) long

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable_analysis_mode(self) -> None:
        """Enable per-head output capture in the attention module."""
        self.attn.analysis_mode = True

    def disable_analysis_mode(self) -> None:
        """Disable per-head output capture (saves memory during training)."""
        self.attn.analysis_mode = False
        self.attn._last_head_outputs = None

    def accumulate(
        self,
        attn_weights_per_batch: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        """Buffer one forward pass for later aggregate analysis.

        Parameters
        ----------
        attn_weights_per_batch : list of Tensor
            ``attn_tensors[-1]`` from the model output.  A list over batch
            elements; each element has shape ``(num_heads, N_b, N_b)``.
        labels : torch.Tensor
            Per-voxel semantic labels, shape ``(N,)`` long, ordered to
            match the concatenated batch voxels.
        """
        self._attn_weights_buffer.append([aw.cpu() for aw in attn_weights_per_batch])
        self._labels_buffer.append(labels.cpu())

        if self.attn.analysis_mode and self.attn._last_head_outputs is not None:
            self._head_outputs_buffer.append(
                [ho.cpu() for ho in self.attn._last_head_outputs]
            )

    def clear(self) -> None:
        """Clear all accumulated buffers."""
        self._attn_weights_buffer.clear()
        self._head_outputs_buffer.clear()
        self._labels_buffer.clear()

    # ------------------------------------------------------------------
    # Analysis 1: weight geometry (no data needed)
    # ------------------------------------------------------------------

    def weight_geometry(self) -> Dict[str, np.ndarray]:
        """Measure max head→class influence from model weights alone.

        Returns
        -------
        dict with keys:
            ``influence``  : (num_heads, num_classes) float array.
                L2 norm of ``W_seg[c,:] @ W_O[:,h*d:(h+1)*d]``.
                Larger = head h *can* affect class c more.
            ``cos_sim``    : (num_heads, num_classes) float array.
                Cosine similarity between the class-direction vector in
                each head's output subspace and the dominant class direction.
                Values near 1 mean monosemantic; many high values mean
                superposition.
            ``norm_ratio`` : (num_heads, num_classes) float array.
                Influence normalised so each class sums to 1 across heads.
                Shows which heads "own" each class.
        """
        # Retrieve weight matrices (handle both MinkowskiLinear and nn.Linear)
        W_O = self._get_weight(self.attn.out_proj)   # (in_channels, embed_dim)
        W_seg = self._get_weight(self.seg_linear)    # (num_classes, num_filters)

        num_classes, num_filters = W_seg.shape
        assert W_O.shape[0] == num_filters, (
            f"out_proj output dim ({W_O.shape[0]}) != "
            f"seg_linear input dim ({num_filters})"
        )

        W_O = W_O.to(torch.float32)
        W_seg = W_seg.to(torch.float32)

        influence = np.zeros((self.num_heads, num_classes), dtype=np.float32)
        cos_sim   = np.zeros_like(influence)

        for h in range(self.num_heads):
            # W_O_h: (num_filters, head_dim) — projects head-h outputs to feature space
            W_O_h = W_O[:, h * self.head_dim : (h + 1) * self.head_dim]
            for c in range(num_classes):
                w_c = W_seg[c, :]  # (num_filters,)
                # Effective direction in head-h's space: (head_dim,)
                eff = w_c @ W_O_h
                influence[h, c] = float(eff.norm())
                # Cosine similarity of W_O_h's column space with w_c
                W_O_h_norm = W_O_h / (W_O_h.norm(dim=0, keepdim=True) + 1e-9)
                projections = (w_c.unsqueeze(0) @ W_O_h_norm).squeeze(0)  # (head_dim,)
                col_norms = w_c.norm() + 1e-9
                cos_sim[h, c] = float(projections.abs().max()) / col_norms

        norm_ratio = influence / (influence.sum(axis=0, keepdims=True) + 1e-9)

        return {
            "influence": influence,
            "cos_sim": cos_sim,
            "norm_ratio": norm_ratio,
        }

    # ------------------------------------------------------------------
    # Analysis 2: attention pattern class affinity (data-dependent)
    # ------------------------------------------------------------------

    def attention_class_affinity(self) -> Dict[str, np.ndarray]:
        """Compute per-head class affinity matrices from buffered data.

        Returns
        -------
        dict with keys:
            ``affinity``        : (num_heads, num_classes, num_classes) float.
                ``M_h[c1,c2] = mean A_h[i,j]`` for ``class(i)=c1, class(j)=c2``.
            ``within_fraction`` : (num_heads, num_classes) float.
                Diagonal of affinity normalised by row sum.  A value near 1
                means head h almost exclusively attends within class c.
            ``entropy``         : (num_heads,) float.
                Shannon entropy of the *flattened* affinity matrix per head
                (higher = more evenly spread = more polysemantic / superposition).
        """
        if not self._attn_weights_buffer:
            raise RuntimeError("No data accumulated. Call .accumulate() first.")

        aff_sum  = np.zeros((self.num_heads, self.num_classes, self.num_classes), dtype=np.float64)
        aff_cnt  = np.zeros_like(aff_sum)

        flat_label_cursor = 0
        labels_cat = torch.cat(self._labels_buffer, dim=0)  # (N_total,)

        for batch_list in self._attn_weights_buffer:
            for aw in batch_list:
                # aw: (num_heads, N_b, N_b)
                N_b = aw.shape[1]
                lbl = labels_cat[flat_label_cursor : flat_label_cursor + N_b].numpy()
                flat_label_cursor += N_b

                for h in range(self.num_heads):
                    A_h = aw[h].numpy().astype(np.float64)  # (N_b, N_b)
                    for c1 in range(self.num_classes):
                        mask_i = np.where(lbl == c1)[0]
                        if len(mask_i) == 0:
                            continue
                        for c2 in range(self.num_classes):
                            mask_j = np.where(lbl == c2)[0]
                            if len(mask_j) == 0:
                                continue
                            block = A_h[np.ix_(mask_i, mask_j)]
                            aff_sum[h, c1, c2] += block.sum()
                            aff_cnt[h, c1, c2] += block.size

        affinity = np.where(aff_cnt > 0, aff_sum / aff_cnt, 0.0).astype(np.float32)

        row_sum = affinity.sum(axis=2, keepdims=True) + 1e-12
        within_fraction = (affinity / row_sum)[:, np.arange(self.num_classes), np.arange(self.num_classes)]
        # within_fraction: (num_heads, num_classes)

        # Entropy per head over the (num_classes × num_classes) affinity distribution
        entropy = np.zeros(self.num_heads, dtype=np.float32)
        for h in range(self.num_heads):
            p = affinity[h].ravel()
            p = p / (p.sum() + 1e-12)
            entropy[h] = float(-np.sum(p * np.log(p + 1e-12)))

        return {
            "affinity": affinity,
            "within_fraction": within_fraction,
            "entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Analysis 3: feature separability (analysis_mode required)
    # ------------------------------------------------------------------

    def feature_separability(self) -> Dict[str, np.ndarray]:
        """Fisher LDA separability of per-head output features.

        Requires the model to have been run with ``analysis_mode=True`` and
        ``accumulate()`` called after each forward pass.

        A high Fisher ratio for (head h, class c) means voxels of class c
        form a tight cluster *in head h's feature subspace* relative to the
        spread of other classes.  Multiple high-ratio classes per head
        indicate superposition.

        Returns
        -------
        dict with keys:
            ``fisher_ratio``    : (num_heads, num_classes) float.
                Between-class / within-class scatter ratio for each
                (head, class) pair.
            ``probe_accuracy``  : (num_heads, num_classes) float.
                One-vs-rest linear probe accuracy per (head, class).  Fit
                using a closed-form ridge classifier for speed.
            ``superposition_score`` : (num_heads,) float.
                Number of classes with ``probe_accuracy > 0.6`` per head,
                normalised by ``num_classes``.  0 = head is silent;
                1 = all classes decodable (max superposition).
        """
        if not self._head_outputs_buffer:
            raise RuntimeError(
                "No head-output data accumulated. "
                "Enable analysis_mode before forward passes and call .accumulate()."
            )

        # Build flat arrays: features (N_total, num_heads, head_dim), labels (N_total,)
        feats_list = []
        for batch_list in self._head_outputs_buffer:
            for ho in batch_list:
                feats_list.append(ho)  # (N_b, num_heads, head_dim)
        feats = torch.cat(feats_list, dim=0).numpy()  # (N, num_heads, head_dim)

        labels = torch.cat(self._labels_buffer, dim=0).numpy()  # (N,)
        assert feats.shape[0] == len(labels), (
            f"Feature count ({feats.shape[0]}) ≠ label count ({len(labels)}). "
            "Check that accumulate() was called consistently."
        )

        fisher_ratio   = np.zeros((self.num_heads, self.num_classes), dtype=np.float32)
        probe_accuracy = np.zeros_like(fisher_ratio)

        for h in range(self.num_heads):
            X = feats[:, h, :]  # (N, head_dim)
            mu_all = X.mean(axis=0)

            for c in range(self.num_classes):
                mask_c   = labels == c
                mask_not = ~mask_c

                if mask_c.sum() < 2 or mask_not.sum() < 2:
                    continue

                X_c   = X[mask_c]    # (n_c, head_dim)
                X_not = X[mask_not]  # (N-n_c, head_dim)
                mu_c   = X_c.mean(axis=0)
                mu_not = X_not.mean(axis=0)

                # Between-class scatter (scalar version: distance of class mean from others)
                sb = float(np.dot(mu_c - mu_not, mu_c - mu_not))

                # Within-class scatter (average per-class variance projected on the
                # between-class direction)
                direction = mu_c - mu_not
                d_norm = direction / (np.linalg.norm(direction) + 1e-9)
                sw_c   = float(np.var(X_c @ d_norm))
                sw_not = float(np.var(X_not @ d_norm))
                sw = sw_c + sw_not + 1e-9

                fisher_ratio[h, c] = sb / sw

                # Ridge probe: one-vs-rest with regularisation λ=1
                probe_accuracy[h, c] = self._ridge_probe_accuracy(
                    X, mask_c.astype(np.float32)
                )

        # Superposition score: fraction of classes decodable from each head
        superposition_score = (probe_accuracy > 0.6).sum(axis=1) / self.num_classes

        return {
            "fisher_ratio": fisher_ratio,
            "probe_accuracy": probe_accuracy,
            "superposition_score": superposition_score,
        }

    # ------------------------------------------------------------------
    # Convenience: pretty-print summary
    # ------------------------------------------------------------------

    def summary(self, results: Optional[Dict] = None) -> str:
        """Return a human-readable summary of analysis results.

        Parameters
        ----------
        results : dict, optional
            If not provided, runs ``weight_geometry()`` automatically.
        """
        if results is None:
            results = self.weight_geometry()

        lines = ["=" * 60, "Attention Head Superposition Analysis", "=" * 60]

        if "influence" in results:
            inf = results["influence"]  # (num_heads, num_classes)
            lines.append("\nWeight Geometry — max head→class influence (L2 norm):")
            header = "Head  |  " + "  ".join(
                f"{self.class_names.get(c, str(c)):>8}" for c in range(inf.shape[1])
            )
            lines.append(header)
            lines.append("-" * len(header))
            for h in range(inf.shape[0]):
                row = f"  {h:2d}  |  " + "  ".join(f"{inf[h,c]:8.4f}" for c in range(inf.shape[1]))
                lines.append(row)

        if "affinity" in results:
            lines.append("\nAttention Within-Class Fraction (diagonal of affinity / row sum):")
            wf = results["within_fraction"]  # (num_heads, num_classes)
            header = "Head  |  " + "  ".join(
                f"{self.class_names.get(c, str(c)):>8}" for c in range(wf.shape[1])
            )
            lines.append(header)
            lines.append("-" * len(header))
            for h in range(wf.shape[0]):
                row = f"  {h:2d}  |  " + "  ".join(f"{wf[h,c]:8.3f}" for c in range(wf.shape[1]))
                lines.append(row)
            ent = results["entropy"]
            lines.append(f"\nPer-head affinity entropy (higher = more polysemantic):")
            for h in range(len(ent)):
                lines.append(f"  Head {h}: {ent[h]:.4f}")

        if "probe_accuracy" in results:
            lines.append("\nFeature Separability — one-vs-rest linear probe accuracy:")
            pa = results["probe_accuracy"]  # (num_heads, num_classes)
            header = "Head  |  " + "  ".join(
                f"{self.class_names.get(c, str(c)):>8}" for c in range(pa.shape[1])
            )
            lines.append(header)
            lines.append("-" * len(header))
            for h in range(pa.shape[0]):
                row = f"  {h:2d}  |  " + "  ".join(f"{pa[h,c]:8.3f}" for c in range(pa.shape[1]))
                lines.append(row)
            ss = results["superposition_score"]
            lines.append(f"\nSuperposition score (fraction of classes decodable per head):")
            for h in range(len(ss)):
                lines.append(f"  Head {h}: {ss[h]:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_weight(layer) -> torch.Tensor:
        """Extract the weight matrix from a MinkowskiLinear or nn.Linear layer."""
        if hasattr(layer, "linear"):
            # MinkowskiLinear wraps an nn.Linear
            return layer.linear.weight.detach().cpu()
        elif hasattr(layer, "weight"):
            return layer.weight.detach().cpu()
        else:
            raise AttributeError(f"Cannot extract weight from {type(layer)}")

    @staticmethod
    def _ridge_probe_accuracy(
        X: np.ndarray, binary_labels: np.ndarray, lam: float = 1.0
    ) -> float:
        """Closed-form ridge regression one-vs-rest probe, returns accuracy.

        Solves  ``w = (X^T X + λI)^{-1} X^T y``  then thresholds at 0.5.
        Fast enough for analysis (no iterative solver needed at typical sizes).
        """
        n, d = X.shape
        # Normalise features for numerical stability
        mu = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        Xn = (X - mu) / std

        if d <= n:
            A = Xn.T @ Xn + lam * np.eye(d)
            w = np.linalg.solve(A, Xn.T @ binary_labels)
        else:
            # Woodbury / dual form when d >> n (rare for head_dim << N)
            A = Xn @ Xn.T + lam * np.eye(n)
            alpha = np.linalg.solve(A, binary_labels)
            w = Xn.T @ alpha

        preds = (Xn @ w > 0.5).astype(np.float32)
        return float((preds == binary_labels).mean())
