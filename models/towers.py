"""
User and item encoder towers for Spotlight.

Both towers share the same architecture pattern:

    categorical fields --> per-field nn.Embedding --> concat
    numeric features   --> Linear projection
                          \\__ concat __/
                                 |
                             MLP (ReLU + dropout)
                                 |
                          Linear -> embedding_dim
                                 |
                        (optional) L2-normalize

The two towers are *separate* networks (no shared weights) but emit vectors into
the *same* embedding space of dimension ``embedding_dim``. Retrieval works
because the contrastive loss pulls a user's embedding toward its positive item's
embedding and pushes it away from in-batch negatives.

L2-normalization is important: when both ``u`` and ``v`` are unit vectors,
``u . v == cos(u, v) in [-1, 1]``, so a plain inner-product FAISS index
(``IndexFlatIP`` / ``IndexIVFFlat`` with metric IP) returns exact cosine-nearest
neighbors.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """
    Encodes (categorical, numeric) raw features into a dense vector.

    Categorical fields each get their own embedding table; numeric features pass
    through a single linear layer. The two are concatenated to form the encoder's
    output, which the tower MLP then consumes.

    Args:
        categorical_cardinalities: vocab size per categorical field.
        n_numeric: number of dense numeric features.
        feature_embedding_dim: width of each categorical embedding.
    """

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        n_numeric: int,
        feature_embedding_dim: int,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=card, embedding_dim=feature_embedding_dim)
                for card in categorical_cardinalities
            ]
        )
        self.n_numeric = n_numeric
        # Project numeric block to the same per-field width for balance.
        self.numeric_proj = (
            nn.Linear(n_numeric, feature_embedding_dim) if n_numeric > 0 else None
        )

        n_cat_fields = len(categorical_cardinalities)
        self.output_dim = n_cat_fields * feature_embedding_dim + (
            feature_embedding_dim if n_numeric > 0 else 0
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat: (B, n_cat_fields) long categorical ids.
            num: (B, n_numeric) float numeric features.

        Returns:
            (B, output_dim) concatenated dense feature vector.
        """
        parts: List[torch.Tensor] = []
        for f, emb in enumerate(self.embeddings):
            parts.append(emb(cat[:, f]))  # (B, feature_embedding_dim)
        if self.numeric_proj is not None:
            parts.append(self.numeric_proj(num))  # (B, feature_embedding_dim)
        return torch.cat(parts, dim=-1)


class _Tower(nn.Module):
    """
    Shared implementation for both towers: FeatureEncoder + MLP -> embedding.

    Args:
        categorical_cardinalities: vocab size per categorical field.
        n_numeric: number of dense numeric features.
        feature_embedding_dim: per-field categorical embedding width.
        hidden: list of MLP hidden layer sizes.
        embedding_dim: output (shared-space) dimensionality.
        dropout: dropout probability between MLP layers.
        l2_normalize: if True, L2-normalize the output embedding.
    """

    def __init__(
        self,
        categorical_cardinalities: Sequence[int],
        n_numeric: int,
        feature_embedding_dim: int,
        hidden: Sequence[int],
        embedding_dim: int,
        dropout: float,
        l2_normalize: bool,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            categorical_cardinalities, n_numeric, feature_embedding_dim
        )
        self.l2_normalize = l2_normalize

        layers: List[nn.Module] = []
        in_dim = self.encoder.output_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small-init the linear/embedding layers for stable cosine training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cat: (B, n_cat_fields) long categorical ids.
            num: (B, n_numeric) float numeric features.

        Returns:
            (B, embedding_dim) embedding, L2-normalized iff ``l2_normalize``.
        """
        x = self.encoder(cat, num)
        emb = self.mlp(x)
        if self.l2_normalize:
            # Unit vectors => inner product equals cosine similarity.
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


class UserTower(_Tower):
    """Encodes user features into the shared embedding space."""


class ItemTower(_Tower):
    """Encodes item features into the shared embedding space."""
