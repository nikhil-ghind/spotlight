# Spotlight — Design Document

## Problem

Large-scale recommendation and retrieval systems must, for any given user,
surface a handful of relevant items out of a catalog of millions in a few
milliseconds. Scoring every (user, item) pair with a heavy cross-attention model
is infeasible at serve time. The standard solution is **retrieval/ranking
decomposition**: a cheap *retrieval* stage narrows millions of items down to a
few hundred candidates, and an expensive *ranking* stage re-scores them.
Spotlight implements the retrieval stage as a **two-tower (dual-encoder)** model.

## Two-tower architecture

There are two independent encoders that map into one shared embedding space of
dimension `d`:

- **User tower** `f_θ`: user features (categorical + numeric) → `u ∈ R^d`.
- **Item tower** `g_φ`: item features (categorical + numeric) → `v ∈ R^d`.

Each tower:

1. Maps every categorical field through its own `nn.Embedding` table.
2. Projects the dense numeric block through a `Linear` layer.
3. Concatenates the pieces and passes them through an MLP (ReLU + dropout).
4. Projects to `d` dimensions and **L2-normalizes** the output.

The relevance of an item to a user is the dot product `s(u, v) = uᵀv`. Because
both vectors are unit-norm, this dot product equals their **cosine similarity**.

The key property that makes this fast: the towers are *decoupled*. The user
tower only sees user features; the item tower only sees item features. So item
embeddings can be precomputed offline for the whole catalog, and at serve time
we only run the (small) user tower once and do a nearest-neighbor lookup.

## In-batch negative sampling (training math)

We want `uᵀv⁺` to be high for true positive pairs and low for everything else.
The "everything else" is the entire catalog, so the exact softmax is

```
P(i⁺ | u) = exp(uᵀv_{i⁺} / τ) / Σ_{j ∈ catalog} exp(uᵀv_j / τ)
```

The denominator (partition function) sums over 50K+ items — far too expensive
per step. **In-batch negative sampling** approximates it using only the items
already in the current minibatch of `B` positive pairs:

```
Given U = [u_1..u_B], V = [v_1..v_B]  with v_i the positive for u_i,
  S = (U Vᵀ) / τ              # (B, B) similarity matrix
  L_u2i = (1/B) Σ_i -log( exp(S_ii) / Σ_j exp(S_ij) )   # cross-entropy, target=i
```

Row `i` is a `B`-way classification: the true class is the diagonal `S_ii`
(item `v_i`), and every off-diagonal `S_ij` (`j ≠ i`) is item `v_j`, a positive
for *some other* user that here serves as a free **negative** for user `i`. One
batch of size `B` therefore yields `B−1` negatives per example with no extra
data loading.

We optimize the **symmetric** loss `L = ½(L_u2i + L_i2u)` (item→user direction
added via the transposed matrix), which is standard for dual encoders and trains
more stably.

`τ` (temperature) sharpens the softmax: small `τ` makes the model very confident
and pushes negatives hard; large `τ` softens gradients. We default to `τ = 0.07`.

## Nightly index refresh

Item features (and the item tower) change over time, so item embeddings must be
recomputed periodically — typically a **nightly batch job**:

```
interaction logs ──► train two-tower ──► embed ALL items (item tower)
                                              │
                                              ▼
                                   build FAISS index (IVFFlat)
                                              │
                                              ▼
                                   persist + atomically deploy
```

At serve time the user tower runs online (per request) while the item side is a
static, pre-built FAISS index. This asymmetry is exactly why two-tower retrieval
scales.

## Evaluation methodology

We use a **leave-one-out** split: for a random subset of users we hold out one
interaction. Evaluation embeds each held-out user, retrieves the top-`max(K)`
items from the FAISS index, and measures:

- **Recall@K** (K = 10, 50): fraction of users whose held-out item is in the
  top-K. With one positive per user this equals Hit-Rate@K.
- **MRR**: mean of `1 / rank(item⁺)`, rewarding placing the true item high.

Evaluating *through the FAISS index* (not via exact full-catalog scoring) means
the reported numbers include any recall loss from the ANN approximation — i.e.
they reflect real serving behavior.

## Tradeoffs and design decisions

- **L2-normalized embeddings + inner product == cosine.** Lets us use a plain IP
  FAISS index and keeps similarities bounded in `[-1, 1]`, which stabilizes the
  temperature-scaled softmax.
- **In-batch negatives** are nearly free but biased toward popular items (popular
  items appear as negatives more often). Larger batches reduce variance and
  improve the softmax approximation; production systems sometimes add a
  popularity-frequency correction (logQ) — omitted here for clarity.
- **IndexIVFFlat vs IndexFlatIP.** Flat is exact but `O(N·d)` per query. IVF
  clusters items into `nlist` cells and scans only `nprobe` of them, giving large
  speedups at a small, tunable recall cost. We expose both; IVF is the default.
- **Nightly refresh** decouples (slow) item embedding from (fast) online serving
  and bounds embedding staleness to one day — a good balance for most catalogs.
- **Decoupled towers** preclude user×item feature crosses at retrieval time; that
  modeling power is deferred to the downstream ranking stage (out of scope here).
