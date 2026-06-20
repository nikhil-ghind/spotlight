"""
TwoTowerModel: wires the user and item towers and implements the in-batch
sampled-softmax (contrastive) training objective.

In-batch negative sampling
---------------------------
Given a batch of B positive pairs, let

    U = [u_1, ..., u_B]   (B, d)  L2-normalized user embeddings
    V = [v_1, ..., v_B]   (B, d)  L2-normalized item embeddings  (v_i positive for u_i)

We form the full B x B similarity matrix and scale by 1/temperature:

    S = (U @ V^T) / tau           (B, B)

Row i is treated as a (B-way) classification problem whose correct class is the
diagonal entry i (the true positive). Every off-diagonal column j != i is a
*negative*: item v_j, which is positive for some *other* user, acts as a sampled
negative for user i. The loss is cross-entropy with targets = arange(B):

    L = (1/B) * sum_i  -log( exp(S[i,i]) / sum_j exp(S[i,j]) )

This is a sampled-softmax approximation of the true softmax over all 50K items:
instead of scoring every item in the catalog (expensive), we approximate the
partition function using the B items already in the batch. Larger batches => more
negatives => a tighter approximation and a stronger learning signal.

We use the *symmetric* form (user->item and item->user) which is standard for
dual encoders and tends to train more stably.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic import FeatureSpec
from models.towers import ItemTower, UserTower


class TwoTowerModel(nn.Module):
    """
    Dual-encoder retrieval model.

    Args:
        spec: feature layout (cardinalities + numeric counts) for both towers.
        embedding_dim: shared embedding-space dimensionality.
        feature_embedding_dim: per-field categorical embedding width.
        tower_hidden: MLP hidden sizes used by both towers.
        dropout: dropout probability inside the towers.
        temperature: softmax temperature tau for the contrastive loss.
        l2_normalize: whether tower outputs are L2-normalized (cosine geometry).
    """

    def __init__(
        self,
        spec: FeatureSpec,
        embedding_dim: int = 64,
        feature_embedding_dim: int = 16,
        tower_hidden: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        temperature: float = 0.07,
        l2_normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.embedding_dim = embedding_dim

        self.user_tower = UserTower(
            categorical_cardinalities=spec.user_categorical_cardinalities,
            n_numeric=spec.user_n_numeric,
            feature_embedding_dim=feature_embedding_dim,
            hidden=tower_hidden,
            embedding_dim=embedding_dim,
            dropout=dropout,
            l2_normalize=l2_normalize,
        )
        self.item_tower = ItemTower(
            categorical_cardinalities=spec.item_categorical_cardinalities,
            n_numeric=spec.item_n_numeric,
            feature_embedding_dim=feature_embedding_dim,
            hidden=tower_hidden,
            embedding_dim=embedding_dim,
            dropout=dropout,
            l2_normalize=l2_normalize,
        )

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    def encode_user(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """Embed a batch of users -> (B, embedding_dim)."""
        return self.user_tower(cat, num)

    def encode_item(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """Embed a batch of items -> (B, embedding_dim)."""
        return self.item_tower(cat, num)

    # ------------------------------------------------------------------ #
    # Training objective
    # ------------------------------------------------------------------ #
    def compute_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Symmetric in-batch contrastive loss with in-batch negatives.

        Args:
            batch: dict with ``user_cat``, ``user_num``, ``item_cat``, ``item_num``
                (as produced by ``collate_pairs``). The i-th item is the positive
                for the i-th user.

        Returns:
            loss: scalar loss tensor.
            metrics: dict with in-batch recall@1, mean positive/negative sim, etc.
        """
        u = self.encode_user(batch["user_cat"], batch["user_num"])  # (B, d)
        v = self.encode_item(batch["item_cat"], batch["item_num"])  # (B, d)

        # Similarity matrix scaled by temperature. Because u, v are unit vectors,
        # logits[i, j] is the cosine similarity between user i and item j over tau.
        logits = (u @ v.t()) / self.temperature  # (B, B)
        b = logits.size(0)
        targets = torch.arange(b, device=logits.device)  # diagonal == positives

        # user->item direction: each user must pick its own item out of the batch.
        loss_u2i = F.cross_entropy(logits, targets)
        # item->user direction: each item must pick its own user (transpose).
        loss_i2u = F.cross_entropy(logits.t(), targets)
        loss = 0.5 * (loss_u2i + loss_i2u)

        with torch.no_grad():
            metrics = self._batch_metrics(logits, targets)
            metrics["loss"] = loss.item()

        return loss, metrics

    @staticmethod
    def _batch_metrics(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Cheap diagnostics computed from the in-batch similarity matrix.

        Args:
            logits: (B, B) temperature-scaled similarity matrix.
            targets: (B,) indices of the true positive per row (the diagonal).

        Returns:
            dict with in-batch recall@1, mean positive similarity, mean negative
            similarity (all in *similarity* units, i.e. logits * temperature is not
            re-applied — these are the scaled logits, fine for monitoring).
        """
        b = logits.size(0)
        preds = logits.argmax(dim=1)
        recall_at_1 = (preds == targets).float().mean().item()

        diag = logits.diagonal()  # positive similarities
        pos_sim = diag.mean().item()

        # Mean of off-diagonal (negative) similarities.
        off_mask = ~torch.eye(b, dtype=torch.bool, device=logits.device)
        neg_sim = logits[off_mask].mean().item()

        return {
            "recall_at_1": recall_at_1,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str, spec: FeatureSpec) -> None:
        """Save both towers + the FeatureSpec + key hyperparameters."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "spec": spec.to_dict(),
                "embedding_dim": self.embedding_dim,
                "temperature": self.temperature,
            },
            path,
        )
        print(f"Two-tower model saved to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu", **overrides) -> "TwoTowerModel":
        """
        Load a model checkpoint written by :meth:`save`.

        Args:
            path: checkpoint path.
            device: device to map tensors onto.
            **overrides: optional kwargs forwarded to ``__init__`` (e.g.
                ``feature_embedding_dim``, ``tower_hidden``) that are not stored in
                the checkpoint and must match the trained architecture.

        Returns:
            A ready-to-use TwoTowerModel.
        """
        ckpt = torch.load(path, map_location=device)
        spec = FeatureSpec.from_dict(ckpt["spec"])
        model = cls(
            spec=spec,
            embedding_dim=ckpt["embedding_dim"],
            temperature=ckpt["temperature"],
            **overrides,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model
