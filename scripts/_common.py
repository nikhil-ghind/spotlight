"""
Shared helpers for the Spotlight CLI scripts.

Keeps config loading, device selection, and bulk item embedding in one place so
``train.py``, ``build_index.py``, ``serve.py``, and the evaluator stay small and
consistent.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import yaml

from data.synthetic import SyntheticDataset
from models.two_tower import TwoTowerModel


def load_config(path: str) -> Dict[str, Any]:
    """Load the YAML config at ``path`` into a nested dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def pick_device(prefer: str = "auto") -> torch.device:
    """
    Choose a torch device.

    Args:
        prefer: "auto", "cpu", "cuda", or "mps".

    Returns:
        a torch.device.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_all_items(
    model: TwoTowerModel,
    data: SyntheticDataset,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Embed the entire item catalog with the item tower (the nightly batch job).

    Args:
        model: a trained TwoTowerModel (will be set to eval()).
        data: dataset providing item feature matrices.
        device: device to run inference on.
        batch_size: items per forward pass.

    Returns:
        (n_items, embedding_dim) float32 L2-normalized item embeddings, row i
        corresponding to catalog item id i.
    """
    model.eval()
    n = data.n_items
    item_cat = torch.from_numpy(np.ascontiguousarray(data.item_cat))
    item_num = torch.from_numpy(np.ascontiguousarray(data.item_num))

    out = np.empty((n, model.embedding_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cat = item_cat[start:end].to(device)
        num = item_num[start:end].to(device)
        emb = model.encode_item(cat, num)
        out[start:end] = emb.cpu().numpy()
    return out


@torch.no_grad()
def embed_users(
    model: TwoTowerModel,
    data: SyntheticDataset,
    user_ids: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Embed a set of users with the user tower.

    Args:
        model: a trained TwoTowerModel.
        data: dataset providing user feature matrices.
        user_ids: (Q,) catalog user ids to embed.
        device: device to run inference on.
        batch_size: users per forward pass.

    Returns:
        (Q, embedding_dim) float32 L2-normalized user embeddings.
    """
    model.eval()
    user_cat = torch.from_numpy(np.ascontiguousarray(data.user_cat))
    user_num = torch.from_numpy(np.ascontiguousarray(data.user_num))

    out = np.empty((len(user_ids), model.embedding_dim), dtype=np.float32)
    for start in range(0, len(user_ids), batch_size):
        end = min(start + batch_size, len(user_ids))
        ids = torch.from_numpy(np.ascontiguousarray(user_ids[start:end])).long()
        cat = user_cat[ids].to(device)
        num = user_num[ids].to(device)
        emb = model.encode_user(cat, num)
        out[start:end] = emb.cpu().numpy()
    return out
