"""
Synthetic interaction-log generator for Spotlight.

The goal is to produce data that *contains learnable signal* so the two-tower
model can actually recover affinity structure end-to-end with zero external
downloads.

How the signal is injected
--------------------------
Every user and every item is assigned a hidden latent vector in R^n_latent. The
*affinity* between a user u and an item i is the dot product of their latent
vectors:

    affinity(u, i) = <z_user[u], z_item[i]>

Observable features (categorical + numeric) are then generated as *noisy linear
projections* of these latent vectors. The towers never see the latent vectors;
they only see the observable features. Because the features carry a (noisy)
shadow of the latent space, a well-trained tower can reconstruct an embedding
space in which the true positive pairs are close — exactly what the contrastive
loss rewards.

Positive interactions are sampled with probability proportional to
softmax(affinity), so popular/compatible pairs co-occur more often, mirroring a
real recommendation log.

Outputs (written by scripts/generate_data.py) are stored in a single ``.npz``
file plus a small JSON metadata sidecar describing feature dimensions.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np


@dataclasses.dataclass
class FeatureSpec:
    """
    Describes the shape of the feature tensors the towers must consume.

    Attributes:
        user_categorical_cardinalities: vocab size for each user categorical field.
        item_categorical_cardinalities: vocab size for each item categorical field.
        user_n_numeric: number of dense numeric user features.
        item_n_numeric: number of dense numeric item features.
    """

    user_categorical_cardinalities: Tuple[int, ...]
    item_categorical_cardinalities: Tuple[int, ...]
    user_n_numeric: int
    item_n_numeric: int

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a plain dict (for JSON metadata)."""
        return {
            "user_categorical_cardinalities": list(self.user_categorical_cardinalities),
            "item_categorical_cardinalities": list(self.item_categorical_cardinalities),
            "user_n_numeric": self.user_n_numeric,
            "item_n_numeric": self.item_n_numeric,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "FeatureSpec":
        """Rebuild a FeatureSpec from its serialized dict form."""
        return cls(
            user_categorical_cardinalities=tuple(d["user_categorical_cardinalities"]),
            item_categorical_cardinalities=tuple(d["item_categorical_cardinalities"]),
            user_n_numeric=int(d["user_n_numeric"]),
            item_n_numeric=int(d["item_n_numeric"]),
        )


@dataclasses.dataclass
class SyntheticDataset:
    """
    Container for a fully-generated synthetic dataset.

    Tensors are stored as numpy arrays; the PyTorch ``Dataset`` in
    ``data/dataset.py`` wraps these without copying.

    Attributes:
        user_cat: (n_users, n_user_cat_fields) int64 categorical ids per user.
        user_num: (n_users, user_n_numeric) float32 numeric features per user.
        item_cat: (n_items, n_item_cat_fields) int64 categorical ids per item.
        item_num: (n_items, item_n_numeric) float32 numeric features per item.
        train_pairs: (n_train, 2) int64 (user_id, item_id) positive pairs.
        val_pairs: (n_val, 2) int64 held-out (user_id, item_id) positive pairs.
        spec: FeatureSpec describing the feature layout.
    """

    user_cat: np.ndarray
    user_num: np.ndarray
    item_cat: np.ndarray
    item_num: np.ndarray
    train_pairs: np.ndarray
    val_pairs: np.ndarray
    spec: FeatureSpec

    @property
    def n_users(self) -> int:
        return int(self.user_cat.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_cat.shape[0])


def _project_latent_to_numeric(
    latent: np.ndarray,
    n_numeric: int,
    rng: np.random.Generator,
    noise: float = 0.3,
) -> np.ndarray:
    """
    Project a latent matrix to noisy dense numeric features.

    A random linear map ``W`` (latent_dim -> n_numeric) plus Gaussian noise gives
    observable features that are a degraded view of the latent space. Features are
    then standardized to roughly zero-mean/unit-variance, which is what the tower
    expects at its numeric input.

    Args:
        latent: (n, latent_dim) hidden vectors.
        n_numeric: desired number of numeric features.
        rng: numpy random generator.
        noise: stddev of additive Gaussian noise (higher => harder task).

    Returns:
        (n, n_numeric) float32 standardized numeric features.
    """
    latent_dim = latent.shape[1]
    w = rng.normal(0.0, 1.0, size=(latent_dim, n_numeric))
    feats = latent @ w
    feats = feats + rng.normal(0.0, noise, size=feats.shape)
    # Standardize per-column.
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (
        feats.std(axis=0, keepdims=True) + 1e-6
    )
    return feats.astype(np.float32)


def _project_latent_to_categorical(
    latent: np.ndarray,
    cardinalities: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Assign each entity a categorical id per field, correlated with its latent.

    For each categorical field we create ``cardinality`` random *prototype*
    vectors in the latent space and assign the id of the nearest prototype (by
    dot product). This makes categories carry latent signal: entities with similar
    latents tend to share categories, just like genres or brands cluster tastes.

    Args:
        latent: (n, latent_dim) hidden vectors.
        cardinalities: vocab size for each categorical field.
        rng: numpy random generator.

    Returns:
        (n, n_fields) int64 category ids.
    """
    n = latent.shape[0]
    latent_dim = latent.shape[1]
    out = np.zeros((n, len(cardinalities)), dtype=np.int64)
    for f, card in enumerate(cardinalities):
        prototypes = rng.normal(0.0, 1.0, size=(card, latent_dim))
        scores = latent @ prototypes.T  # (n, card)
        # Add noise so it is not a deterministic function of the latent.
        scores = scores + rng.normal(0.0, 0.5, size=scores.shape)
        out[:, f] = scores.argmax(axis=1)
    return out


def generate(
    n_users: int,
    n_items: int,
    n_interactions: int,
    n_latent: int,
    user_categorical_cardinalities: Tuple[int, ...],
    item_categorical_cardinalities: Tuple[int, ...],
    user_n_numeric: int,
    item_n_numeric: int,
    val_frac: float,
    seed: int,
) -> SyntheticDataset:
    """
    Generate a complete synthetic dataset with learnable user/item affinity.

    Args:
        n_users: number of users.
        n_items: number of items (catalog size).
        n_interactions: number of positive interaction logs to sample.
        n_latent: dimensionality of the hidden affinity space.
        user_categorical_cardinalities: vocab sizes for user categorical fields.
        item_categorical_cardinalities: vocab sizes for item categorical fields.
        user_n_numeric: number of dense numeric user features.
        item_n_numeric: number of dense numeric item features.
        val_frac: fraction of *users* whose last interaction is held out for eval.
        seed: RNG seed for reproducibility.

    Returns:
        A populated SyntheticDataset.
    """
    rng = np.random.default_rng(seed)

    # 1. Hidden latent vectors. Item popularity is encoded by a per-item bias so
    #    that some items are globally more likely (a realistic long-tail).
    z_user = rng.normal(0.0, 1.0, size=(n_users, n_latent))
    z_item = rng.normal(0.0, 1.0, size=(n_items, n_latent))
    item_popularity = rng.normal(0.0, 1.0, size=(n_items,))

    # 2. Observable features are noisy projections of the latents.
    user_num = _project_latent_to_numeric(z_user, user_n_numeric, rng)
    item_num = _project_latent_to_numeric(z_item, item_n_numeric, rng)
    user_cat = _project_latent_to_categorical(
        z_user, user_categorical_cardinalities, rng
    )
    item_cat = _project_latent_to_categorical(
        z_item, item_categorical_cardinalities, rng
    )

    # 3. Sample positive interactions. For a randomly chosen user, the item is
    #    drawn from softmax(affinity + popularity) over a candidate sub-sample
    #    (full softmax over 50K items per draw would be needlessly expensive).
    user_ids = rng.integers(0, n_users, size=n_interactions)
    item_ids = np.empty(n_interactions, dtype=np.int64)
    candidate_pool = min(n_items, 512)  # items scored per interaction draw

    # Pre-scale latents for sharper affinity (controls how peaked the softmax is).
    affinity_scale = 1.0 / np.sqrt(n_latent)

    for k in range(n_interactions):
        u = user_ids[k]
        cand = rng.integers(0, n_items, size=candidate_pool)
        logits = affinity_scale * (z_item[cand] @ z_user[u]) + item_popularity[cand]
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        item_ids[k] = cand[rng.choice(candidate_pool, p=probs)]

    pairs = np.stack([user_ids.astype(np.int64), item_ids], axis=1)

    # 4. Leave-one-out style split: hold out one interaction for a random subset
    #    of users. This mimics "predict the user's next item" evaluation.
    train_pairs, val_pairs = _leave_one_out_split(pairs, val_frac, rng)

    spec = FeatureSpec(
        user_categorical_cardinalities=tuple(user_categorical_cardinalities),
        item_categorical_cardinalities=tuple(item_categorical_cardinalities),
        user_n_numeric=user_n_numeric,
        item_n_numeric=item_n_numeric,
    )

    return SyntheticDataset(
        user_cat=user_cat,
        user_num=user_num,
        item_cat=item_cat,
        item_num=item_num,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        spec=spec,
    )


def _leave_one_out_split(
    pairs: np.ndarray,
    val_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hold out exactly one interaction for a random ``val_frac`` of users.

    Every held-out user keeps the rest of their interactions in train, so the
    model has seen the user but not the specific positive it is evaluated on.

    Args:
        pairs: (n, 2) (user_id, item_id) positive pairs.
        val_frac: fraction of distinct users to hold out one interaction for.
        rng: numpy random generator.

    Returns:
        (train_pairs, val_pairs) numpy arrays.
    """
    n = pairs.shape[0]
    users = pairs[:, 0]
    unique_users = np.unique(users)
    n_val_users = int(len(unique_users) * val_frac)
    val_users = set(rng.choice(unique_users, size=n_val_users, replace=False).tolist())

    is_val = np.zeros(n, dtype=bool)
    seen: Dict[int, bool] = {}
    # Iterate in a shuffled order so the held-out interaction is random per user.
    order = rng.permutation(n)
    for idx in order:
        u = int(users[idx])
        if u in val_users and u not in seen:
            is_val[idx] = True
            seen[u] = True

    return pairs[~is_val], pairs[is_val]


def save(dataset: SyntheticDataset, path: str) -> None:
    """
    Persist a SyntheticDataset to a compressed ``.npz`` archive.

    Args:
        dataset: the dataset to save.
        path: output path ending in ``.npz``.
    """
    spec = dataset.spec
    np.savez_compressed(
        path,
        user_cat=dataset.user_cat,
        user_num=dataset.user_num,
        item_cat=dataset.item_cat,
        item_num=dataset.item_num,
        train_pairs=dataset.train_pairs,
        val_pairs=dataset.val_pairs,
        user_categorical_cardinalities=np.asarray(
            spec.user_categorical_cardinalities, dtype=np.int64
        ),
        item_categorical_cardinalities=np.asarray(
            spec.item_categorical_cardinalities, dtype=np.int64
        ),
        user_n_numeric=np.int64(spec.user_n_numeric),
        item_n_numeric=np.int64(spec.item_n_numeric),
    )


def load(path: str) -> SyntheticDataset:
    """
    Load a SyntheticDataset previously written by :func:`save`.

    Args:
        path: path to the ``.npz`` archive.

    Returns:
        A reconstructed SyntheticDataset.
    """
    z = np.load(path)
    spec = FeatureSpec(
        user_categorical_cardinalities=tuple(
            int(x) for x in z["user_categorical_cardinalities"]
        ),
        item_categorical_cardinalities=tuple(
            int(x) for x in z["item_categorical_cardinalities"]
        ),
        user_n_numeric=int(z["user_n_numeric"]),
        item_n_numeric=int(z["item_n_numeric"]),
    )
    return SyntheticDataset(
        user_cat=z["user_cat"],
        user_num=z["user_num"],
        item_cat=z["item_cat"],
        item_num=z["item_num"],
        train_pairs=z["train_pairs"],
        val_pairs=z["val_pairs"],
        spec=spec,
    )
