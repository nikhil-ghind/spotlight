"""
PyTorch Dataset and collate utilities for two-tower training.

A single training example is one *positive* (user, item) pair. We deliberately do
NOT materialize negatives here — negatives are formed implicitly at loss time
from the other items in the same batch (in-batch negative sampling). This keeps
the dataset tiny and the data loader fast: every other positive item in a batch
of size B serves as a negative for a given user, giving B-1 free negatives per
example with no extra I/O.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic import SyntheticDataset


class InteractionPairDataset(Dataset):
    """
    Yields per-example feature tensors for one positive (user, item) pair.

    Each ``__getitem__`` returns the user's features and the *positive* item's
    features. The collate function batches these; in-batch negatives are derived
    downstream inside ``TwoTowerModel.compute_loss``.

    Args:
        data: the in-memory SyntheticDataset.
        split: "train" or "val" — selects which pair list to iterate.
    """

    def __init__(self, data: SyntheticDataset, split: str = "train"):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self._data = data
        self._pairs = data.train_pairs if split == "train" else data.val_pairs

        # Cache feature matrices as torch tensors once (shared, zero-copy views).
        self._user_cat = torch.from_numpy(np.ascontiguousarray(data.user_cat))
        self._user_num = torch.from_numpy(np.ascontiguousarray(data.user_num))
        self._item_cat = torch.from_numpy(np.ascontiguousarray(data.item_cat))
        self._item_num = torch.from_numpy(np.ascontiguousarray(data.item_num))

    def __len__(self) -> int:
        return int(self._pairs.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_id = int(self._pairs[idx, 0])
        item_id = int(self._pairs[idx, 1])
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "user_cat": self._user_cat[user_id],
            "user_num": self._user_num[user_id],
            "item_cat": self._item_cat[item_id],
            "item_num": self._item_num[item_id],
        }


def collate_pairs(batch: list) -> Dict[str, torch.Tensor]:
    """
    Stack a list of per-example dicts into batched tensors.

    Args:
        batch: list of dicts produced by ``InteractionPairDataset.__getitem__``.

    Returns:
        A dict with batched tensors:
            user_id:  (B,) long
            item_id:  (B,) long
            user_cat: (B, n_user_cat_fields) long
            user_num: (B, user_n_numeric) float
            item_cat: (B, n_item_cat_fields) long
            item_num: (B, item_n_numeric) float
    """
    return {
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "item_id": torch.stack([b["item_id"] for b in batch]),
        "user_cat": torch.stack([b["user_cat"] for b in batch]),
        "user_num": torch.stack([b["user_num"] for b in batch]),
        "item_cat": torch.stack([b["item_cat"] for b in batch]),
        "item_num": torch.stack([b["item_num"] for b in batch]),
    }
