# Training iterative Graph-SPICE

Iterative Graph-SPICE (`provider: graph_spice_iter`) clusters voxels into
fragments with an attention UResNet embedder. With more than one pass, each
later pass re-runs the embedder with its decoder attention masked by the
clusters from the previous pass, then re-scores the graph edges.

| Template | Passes | Attention feedback |
|---|---|---|
| [graph_spice_iter_1pass.yaml](graph_spice_iter_1pass.yaml) | 1 | none (baseline) |
| [graph_spice_iter_2pass.yaml](graph_spice_iter_2pass.yaml) | 2 | one |

The two templates differ only in `n_iterations` and `use_attention`, so they
form a clean ablation pair. Both use the generic MPV/MPR schema and a radius
edge graph, and use the true semantics to keep low-energy deposits out of
the network.

## Running a job

Fill in the data and output locations from the command line rather than
editing the template:

```bash
python3 bin/run.py -c config/graph_spice/graph_spice_iter_2pass.yaml \
  --world-size 4 \
  --set io.loader.dataset.file_keys=/path/to/train.root \
  --log-dir /path/to/out/loss \
  --weight-prefix /path/to/out/weight/snapshot
```

- `io.loader.minibatch_size` is per GPU. The templates use 16, so 4 GPUs give
  64 events per optimizer step before `train.accum_steps` (4) accumulates
  them into 256.
- Resume an interrupted run with `--resume` (same log dir and weight prefix).
- `--set key.path=value` overrides any key, e.g.
  `--set model.modules.graph_spice.n_iterations=4`.
- Turn on W&B with `--set base.metrics.wandb.project=<project>`, and
  optionally `name`, `tags` and `run_dir` under the same key. W&B and
  TensorBoard cannot both be on.

On S3DF, run inside the SPINE container:

```bash
singularity exec --nv \
  --bind /sdf/data/neutrino/,/sdf/home/<u>/<user>/ \
  /sdf/group/neutrino/images/develop.sif \
  python3 bin/run.py -c ...
```

## What to set in the config

All model options live under `model.modules.graph_spice`, and all loss
options live under `model.modules.graph_spice_loss`.

### Inputs

`network_input` takes `data`, `seg_label` and `clust_label`. `loss_input`
takes `seg_label` and `clust_label`, plus `ppn_label` when the point
proposal head is on.

| Key | Template | Meaning |
|---|---|---|
| `shapes` | `[shower, track, michel, delta]` | Classes that are clustered. Low-energy deposits are left out. |
| `filter_input` | `true` | Embed only voxels whose semantics are in `shapes`. `false` embeds every voxel and trains the semantic head on all 5 classes (needs `predict_semantics`). |
| `use_pred_sem_filter` | `false` | After the warmup, build the per-shape graphs from predicted semantics instead of `seg_label`. |
| `pred_sem_filter_warmup_steps` | `20000` | Forward passes that use the true labels before switching to predictions. |

### Iterations

| Key | Template | Meaning |
|---|---|---|
| `n_iterations` | `1` / `2` | Number of embedder passes. 1 is the baseline, 2 adds one feedback, and more is a convergence study. |
| `use_attention` | `false` / `true` | Mask the decoder attention with the previous clusters. `false` does not build the attention block. |
| `use_confidence_mask` | `false` | `true` scales the mask by per-voxel edge confidence; `false` applies a hard mask. |
| `cluster_semantic_pool` | `false` | Single pass only: replace the semantics with per-cluster vote fractions. |
| `track_iter_metrics` | `true` | Log ARI, purity, efficiency, SBD, cohesion and mutual information for each pass as `<metric>_iter<k>`. |
| `log_step` | `8` | How often those metrics are computed. Keep it equal to `base.log_step`. |
| `make_clusters` | `false` | Also return the cluster index lists. The full-chain provider turns this on. |

### Embedder (`embedder`, `embedder.uresnet_attn`)

| Key | Template | Meaning |
|---|---|---|
| `predict_semantics`, `num_classes` | `true`, `5` | Per-voxel semantic head on each pass. |
| `filters`, `depth`, `reps`, `input_kernel` | `64, 5, 2, 5` | UResNet backbone. `filters` must equal `kernel.num_features`. |
| `num_input` | `4` | 1 charge feature plus 3 normalized coordinates. |
| `spatial_size` | `768` | Image size in voxels, used to normalize the coordinates. Use the detector size (for example 6144 for ICARUS). |
| `embed_dim`, `num_heads`, `attn_gain` | `256, 8, 0.5` | Attention block. |
| `use_serial_attn` | `false` | `false`: full 3D attention per event (`ClusterAwareAttn`). `true`: 1D windowed attention along a Hilbert curve, with memory proportional to N × `attn_win_size`. |
| `attn_mode` | `boundary` | `boundary` lets voxels attend across cluster boundary regions; `in_cluster` restricts attention to the same cluster. |
| `use_hard_mask` | `true` | Explicit {0, −inf} mask. Ignored by the serialized path. |
| `attn_win_size`, `attn_order` | `256`, `hilbert` | Serialized attention only. |
| `checkpoint_attn` | `false` | Gradient-checkpoint the attention, trading about 2× attention compute for less memory. |

Off by default: `dropout` (UResNet dropout) and `norm_layer: layer_norm`.

### Edges (`kernel`, `edge_projection`, `constructor`)

| Key | Template | Meaning |
|---|---|---|
| `kernel.provider` | `bilinear` | Edge scorer. Options: `bilinear_out` (more than 1 adds an MLP head of width `mlp_hidden`) and `use_spatial_dist` with `spatial_dist_scale`. |
| `edge_projection` | absent | `{n_layers, hidden_dim}` MLP applied to the node features before the kernel. |
| `constructor.graph` | `radius`, `r: 1.9`, `max_num_neighbors: 32` | Candidate edges. `r` = 1.9 voxels covers all 26 lattice neighbors, so the cap is never hit. Other providers: `knn` (`k`) and `radius_topk` (`k`, `r`: the k nearest within r). |
| `constructor.edge_threshold` | `0.1` | Inverted edge score above which an edge is kept. |
| `constructor.min_size`, `orphan` | `3`, radius 1.9 | Clusters smaller than `min_size` are merged into touching clusters. |
| `constructor.union_graph` | absent | `{spatial, feature}` graphs whose union replaces `graph` in refinement passes once `feature_graph_warmup_steps` is reached. |

### Loss (`graph_spice_loss`)

| Key | Template | Meaning |
|---|---|---|
| `loss` | `binary_log_dice_ce` | Edge loss. `edge_loss: {hard_negative_mining: true}` is available. |
| `w_edge`, `w_seg_iter0`, `w_seg_iter_last` | `1.0, 0.3, 0.5` | Weights for the edge loss and for the semantic loss on the first and last passes. |
| `w_contrastive`, `contrastive_loss` | `0` | Optional contrastive loss on the embedding. |
| `evaluate_clustering_metrics` | `true` | Log the final-pass ARI, purity, efficiency and SBD. |

## Experimental options

These options are off by default; turning them on changes nothing else in a
config.

- **Point proposals on the Graph-SPICE backbone.** Add `ppn: {...}` (a PPN
  block) to `graph_spice` and set `ppn_source: iter0 | last` to pick which
  pass feeds it. Add `ppn_loss: {...}` and `w_ppn` to the loss block, and
  `ppn_label` to the loader schema and `loss_input`.
- **Replacing UResNet(-PPN) in the full chain.** Use a single stage that
  publishes `seg_pred`, the fragments and, with `ppn`, `ppn_points`:

  ```yaml
  chain:
    stages:
      - name: semantic_fragmentation
        provider: graph_spice_iter
        uses: graph_spice        # a graph_spice_iter model block
        loss: graph_spice_loss
  ```

  The stage forces `filter_input: false`. If you leave out `ppn`, GrapPA
  needs `add_points`, `add_local_dirs` and `add_local_dedxs` set to `false`
  (with `node_feats` reduced to match), and the interaction `orient` task
  must be dropped; otherwise GrapPA falls back to truth endpoints. The stage
  does not support upstream deghosting yet.
