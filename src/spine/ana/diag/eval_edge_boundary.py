"""
Analysis script: boundary vs interior edge accuracy.

Plug into SPINE via the 'ana' config section:

  ana:
    scripts:
      - module: eval_edge_boundary
        edge_boundary:
          invert: true       # must match model's invert setting
          threshold: 0.5
          output_file: /path/to/edge_boundary_acc.csv

"""

import csv
import os
import numpy as np


class EdgeBoundaryAccuracy:
    """Computes and logs boundary vs interior edge accuracy per batch."""

    name = "edge_boundary"

    def __init__(self, invert=True, threshold=0.5, output_file=None):
        self.invert    = invert
        self.threshold = threshold
        self.output_file = output_file

        # Running accumulators
        self._accs      = []
        self._accs_b    = []
        self._accs_i    = []
        self._n_b       = []
        self._n_i       = []

        self._writer = None
        self._fh     = None
        if output_file:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            self._fh     = open(output_file, "w", newline="")
            self._writer = csv.writer(self._fh)
            self._writer.writerow([
                "batch", "edge_acc", "edge_acc_boundary", "edge_acc_interior",
                "n_boundary", "n_interior", "boundary_fraction"
            ])

    # ------------------------------------------------------------------
    def __call__(self, data):
        ep = data.get("edge_prob")
        el = data.get("edge_label")

        if ep is None or el is None:
            return   # clust_label not provided — skip silently

        # TensorBatch → flat CPU tensor
        if hasattr(ep, "tensor"):
            ep = ep.tensor
        if hasattr(el, "tensor"):
            el = el.tensor

        import torch
        ep = ep.float().cpu()
        el = el.float().cpu()

        # edge_label semantics (always, regardless of invert):
        #   1 = same cluster (connect)   0 = different cluster (cut/boundary)
        # edge_prob semantics with invert=True:
        #   HIGH prob → model predicts CUT   (pred=1 when ep >= threshold)
        #   LOW  prob → model predicts CONNECT
        # So pred=1 means "cut predicted" while el=1 means "connect truth" —
        # they are on opposite scales when invert=True.
        # Normalise pred to connect-positive (1=connect, 0=cut) before comparing.
        pred  = (ep >= self.threshold).long()  # 1 = cut predicted (invert=True)
        truth = el.long()                       # 1 = same cluster (connect)

        # Boundary = different-cluster edges = el==0, regardless of invert
        boundary_mask = (truth == 0)
        interior_mask = (truth == 1)

        # pred_connect: 1 = connect predicted, 0 = cut predicted
        pred_connect = (1 - pred) if self.invert else pred
        correct = (pred_connect == truth).float()
        acc   = correct.mean().item()
        acc_b = correct[boundary_mask].mean().item() if boundary_mask.any() else float("nan")
        acc_i = correct[interior_mask].mean().item() if interior_mask.any() else float("nan")
        n_b   = int(boundary_mask.sum())
        n_i   = int(interior_mask.sum())
        frac  = n_b / (n_b + n_i + 1e-8)

        self._accs.append(acc)
        self._accs_b.append(acc_b)
        self._accs_i.append(acc_i)
        self._n_b.append(n_b)
        self._n_i.append(n_i)

        batch_id = len(self._accs)
        if self._writer:
            self._writer.writerow([batch_id, acc, acc_b, acc_i, n_b, n_i, frac])
            self._fh.flush()

    # ------------------------------------------------------------------
    def close(self):
        if self._fh:
            self._fh.close()
        if not self._accs:
            return
        print("\n=== EdgeBoundaryAccuracy summary ===")
        print(f"  edge_acc          = {np.nanmean(self._accs):.4f}")
        print(f"  edge_acc_boundary = {np.nanmean(self._accs_b):.4f}"
              f"  (mean n_boundary/batch = {np.mean(self._n_b):.0f})")
        print(f"  edge_acc_interior = {np.nanmean(self._accs_i):.4f}"
              f"  (mean n_interior/batch = {np.mean(self._n_i):.0f})")
        frac = np.mean(self._n_b) / (np.mean(self._n_b) + np.mean(self._n_i) + 1e-8)
        print(f"  boundary fraction = {frac:.3f}")
