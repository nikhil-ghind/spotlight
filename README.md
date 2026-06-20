# Spotlight

A two-tower (dual-encoder) recommendation/retrieval system in Python/PyTorch. A
user tower and an item tower encode features into a shared embedding space and
are trained end-to-end with **in-batch negative sampling** (a sampled-softmax
contrastive loss). The item tower embeds a catalog of **50K+ items** into vectors
that are refreshed nightly into a **FAISS** index, enabling **sub-millisecond**
approximate-nearest-neighbor (ANN) candidate retrieval at serve time. Quality is
measured with **Recall@10**, **Recall@50**, and **MRR**.

## What is two-tower retrieval?

Scoring every user against millions of items with one heavy model is too slow
online. Two-tower retrieval splits the work: a user tower and an item tower each
map features into the *same* embedding space, relevance is just a dot product,
and item embeddings are precomputed so serving is one user-tower forward pass
plus an ANN lookup.

```
┌────────────────────────────────────────────────────────────────────┐
│                    Spotlight two-tower retrieval                     │
│                                                                      │
│  User features                              Item features            │
│  (country, age,            ┌─ shared ─┐     (category, brand,        │
│   numeric...)              │ embedding │      numeric...)            │
│       │                    │   space   │           │                 │
│       ▼                    │   (R^d)   │           ▼                 │
│  ┌──────────┐              │           │      ┌──────────┐           │
│  │ UserTower│ ──► u ───────┼──► u·v ◄──┼───── v ◄──│ItemTower│       │
│  │  f_θ     │   (L2-norm)  │  = cosine │   (L2-norm)│  g_φ    │       │
│  └──────────┘              └───────────┘            └──────────┘     │
│                                                                      │
│  TRAINING: in-batch negatives — items of other users in the batch    │
│            act as negatives;  L = cross-entropy(U·Vᵀ / τ, diag)       │
│                                                                      │
│  NIGHTLY:  item tower embeds ALL 50K+ items ──► FAISS index           │
│                                                                      │
│  SERVING:  user emb ──► FAISS ANN search ──► top-k recommended items  │
│            (sub-millisecond)                                         │
└────────────────────────────────────────────────────────────────────┘
```

### Nightly refresh pipeline

```
  interaction logs
        │
        ▼
   ┌─────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  train  │ ──► │ embed ALL items  │ ──► │ build FAISS index │
   │ 2-tower │     │  (item tower)    │     │   (IVFFlat / IP)  │
   └─────────┘     └──────────────────┘     └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │ persist + deploy │
                                            │  items.faiss     │
                                            └──────────────────┘
```

## Project Structure

```
spotlight/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── synthetic.py         # Synthetic interaction-log generator (learnable signal)
│   └── dataset.py           # PyTorch Dataset + collate for positive pairs
├── models/
│   ├── towers.py            # UserTower / ItemTower: feature embeddings + MLP -> L2 emb
│   └── two_tower.py         # TwoTowerModel + in-batch sampled-softmax loss
├── index/
│   └── faiss_index.py       # FAISS IndexFlatIP / IndexIVFFlat build/save/load/query
├── scripts/
│   ├── generate_data.py     # CLI: write synthetic logs to data/artifacts/*.npz
│   ├── train.py             # CLI: train with in-batch negatives, save checkpoint
│   ├── build_index.py       # CLI: nightly refresh — embed all items, build FAISS index
│   └── serve.py             # CLI: user id -> ANN search -> top-k (sub-ms timing)
├── eval/
│   ├── metrics.py           # recall_at_k (10, 50) and mrr
│   └── evaluate.py          # CLI: evaluate held-out positives through the index
├── PLAN.md                  # Design doc
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

All commands below are run from the `spotlight/` directory so that `python -m`
resolves the package imports (`from models...`, `from data...`, `from index...`).

### 2. Generate synthetic data

```bash
python -m scripts.generate_data --config configs/config.yaml
```

Expected output:
```
Generating synthetic data: 20000 users, 50000 items, 400000 interactions (seed=42)...
Wrote data/artifacts/synthetic.npz  | train_pairs=358000 val_pairs=42000 | 6.3s
```

### 3. Train the two-tower model

```bash
python -m scripts.train \
    --config configs/config.yaml \
    --data data/artifacts/synthetic.npz
```

Expected output:
```
Training on cpu for 5 epoch(s) | batch_size=1024 (=> 1023 in-batch negatives/example)
  step     50 | loss=6.4123 | ib_recall@1=0.012 | pos_sim=0.118 neg_sim=0.041
  step    100 | loss=5.2870 | ib_recall@1=0.083 | pos_sim=0.341 neg_sim=0.022
  step    200 | loss=3.9714 | ib_recall@1=0.241 | pos_sim=0.512 neg_sim=-0.004
Epoch   1 | val_loss=3.6021 | val_ib_recall@1=0.288 | time=58.4s
Epoch   2 | val_loss=2.9447 | val_ib_recall@1=0.402 | time=57.9s
Epoch   3 | val_loss=2.5318 | val_ib_recall@1=0.486 | time=58.1s
Epoch   4 | val_loss=2.2904 | val_ib_recall@1=0.535 | time=58.0s
Epoch   5 | val_loss=2.1657 | val_ib_recall@1=0.561 | time=58.2s
Two-tower model saved to checkpoints/two_tower_best.pt
Done. Best val_loss=2.1657 | checkpoint: checkpoints/two_tower_best.pt
```

The loss falls and `pos_sim` (mean similarity of true pairs) rises well above
`neg_sim` (mean similarity of in-batch negatives) — the model is learning to
separate positives from negatives in the shared space.

### 4. Build the FAISS index (nightly refresh)

```bash
python -m scripts.build_index \
    --config configs/config.yaml \
    --data data/artifacts/synthetic.npz \
    --checkpoint checkpoints/two_tower_best.pt
```

Expected output:
```
Embedding all 50000 items with the item tower ...
  embedded 50000 items in 1.8s (dim=64)
Building FAISS index (type=ivf, metric=ip) ...
FAISS index (50000 vectors) saved to index/artifacts/items.faiss
Nightly refresh complete | build=0.9s | index=index/artifacts/items.faiss
```

### 5. Serve recommendations for a user

```bash
python -m scripts.serve \
    --config configs/config.yaml \
    --data data/artifacts/synthetic.npz \
    --checkpoint checkpoints/two_tower_best.pt \
    --index index/artifacts/items.faiss \
    --user-id 1234 --k 10
```

Expected output:
```
User 1234 | index=50000 items | top-10:
   1. item  40213   cos_sim=0.8132
   2. item  10887   cos_sim=0.7945
   3. item  29551   cos_sim=0.7710
   ...
  10. item   6042   cos_sim=0.6988

ANN search latency: 0.214 ms/query (mean over 200 trials, nprobe=16)
```

### 6. Evaluate retrieval quality

```bash
python -m eval.evaluate \
    --config configs/config.yaml \
    --data data/artifacts/synthetic.npz \
    --checkpoint checkpoints/two_tower_best.pt \
    --index index/artifacts/items.faiss
```

Expected output:
```
=== Retrieval evaluation (held-out positives) ===
  Recall@10  : 0.2417
  Recall@50  : 0.4583
  MRR        : 0.1326

Batched ANN throughput: 0.0061 ms/query over 42000 queries (nprobe=16)
```

## How in-batch negative sampling works

We want `uᵀv⁺` high for true positives and low for everything else. The "true"
objective is a softmax over the entire 50K-item catalog:

```
P(i⁺ | u) = exp(uᵀv_{i⁺} / τ) / Σ_{j ∈ catalog} exp(uᵀv_j / τ)
```

The denominator is too expensive to compute every step. **In-batch negative
sampling** approximates it using only the items already in the minibatch. For a
batch of `B` positive pairs with user matrix `U` and item matrix `V`:

```
S = (U Vᵀ) / τ                                        # (B, B) similarity matrix
L_u2i = (1/B) Σ_i  -log( exp(S_ii) / Σ_j exp(S_ij) )  # cross-entropy, target = i
L     = ½ (L_u2i + L_i2u)                             # symmetric (item->user too)
```

Row `i` is a `B`-way classification whose correct class is the diagonal `S_ii`
(item `v_i`). Every off-diagonal `v_j` (`j ≠ i`) is a positive for *another* user
and serves as a **free negative** for user `i` — `B−1` negatives per example with
no extra data loading. Larger batches give more negatives and a tighter softmax
approximation. The implementation is `TwoTowerModel.compute_loss` in
`models/two_tower.py`.

## Configuration Reference

All hyperparameters live in `configs/config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `data` | `n_users` | `20000` | Number of distinct users |
| `data` | `n_items` | `50000` | Catalog size (50K+ items) |
| `data` | `n_interactions` | `400000` | Number of positive interaction logs |
| `data` | `n_latent` | `16` | Hidden affinity dim used to inject learnable signal |
| `data` | `val_frac` | `0.1` | Fraction of users with one held-out interaction |
| `model` | `embedding_dim` | `64` | Shared embedding-space dimension (`d`) |
| `model` | `feature_embedding_dim` | `16` | Per categorical-field embedding width |
| `model` | `tower_hidden` | `[256, 128]` | MLP hidden layer sizes per tower |
| `model` | `dropout` | `0.1` | Dropout inside the towers |
| `model` | `l2_normalize` | `true` | L2-normalize outputs (dot product = cosine) |
| `train` | `batch_size` | `1024` | Batch size (= 1 + in-batch negatives/example) |
| `train` | `lr` | `0.001` | AdamW learning rate |
| `train` | `weight_decay` | `1e-5` | AdamW weight decay |
| `train` | `epochs` | `5` | Training epochs |
| `train` | `temperature` | `0.07` | Softmax temperature `τ` for the contrastive loss |
| `train` | `grad_clip` | `5.0` | Gradient-norm clip |
| `train` | `seed` | `42` | Random seed |
| `index` | `type` | `ivf` | `flat` (exact IndexFlatIP) or `ivf` (IndexIVFFlat) |
| `index` | `nlist` | `256` | IVF Voronoi cells (clusters) |
| `index` | `nprobe` | `16` | IVF cells visited per query (recall/latency knob) |
| `index` | `metric` | `ip` | `ip` (inner product = cosine on unit vectors) or `l2` |
| `eval` | `ks` | `[10, 50]` | Recall@K cutoffs to report |

## Evaluation

`eval/metrics.py` implements the metrics; `eval/evaluate.py` runs them against
held-out positives by querying the FAISS index with trained user embeddings.

| Metric | What it measures | How it is computed |
|--------|------------------|--------------------|
| **Recall@10** | Fraction of held-out users whose true next item is in the top-10 retrieved. | Embed each held-out user, ANN-search the item index for the top-50, check membership in the top-10. |
| **Recall@50** | Same, with a top-50 cutoff (a more forgiving candidate-set quality measure). | As above, checking membership in the full top-50. |
| **MRR** | Mean of `1 / rank(item⁺)` — rewards ranking the true item high, not just retrieving it. | First (best) rank of the held-out item in the retrieved list, reciprocated; misses score 0. |

With one held-out positive per user, Recall@K equals Hit-Rate@K. Evaluating
*through the FAISS index* means the numbers include any recall loss from the IVF
ANN approximation, so they reflect real serving behavior.

## Key Design Decisions

- **L2-normalized embeddings + dot product = cosine.** Unit vectors make
  `uᵀv = cos(u, v) ∈ [-1, 1]`, so a plain inner-product FAISS index returns exact
  cosine neighbors and similarities stay bounded, stabilizing the temperature
  softmax.
- **Temperature `τ`.** Scales logits before the softmax; small `τ` (0.07) sharpens
  the distribution and pushes negatives harder, which is important when all
  similarities are squeezed into `[-1, 1]` by normalization.
- **In-batch negatives** give `B−1` negatives per example at zero extra I/O cost;
  larger batches yield more negatives and a tighter softmax approximation. The
  symmetric (user→item + item→user) loss trains more stably.
- **IndexIVFFlat vs IndexFlatIP.** Flat is exact but `O(N·d)` per query; IVF
  clusters items into `nlist` cells and scans only `nprobe`, trading a little
  recall for a large speedup. `nprobe` is the live recall/latency dial. Both are
  available via `index.type`.
- **Nightly refresh.** Decouples slow full-catalog item embedding from fast online
  serving and bounds embedding staleness to one day — the right balance for most
  catalogs.
- **Decoupled towers.** No user×item feature crosses at retrieval time (that power
  is deferred to a downstream ranking stage), which is exactly what makes
  precomputed item embeddings + ANN possible.
