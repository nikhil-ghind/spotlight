"""
FAISS approximate-nearest-neighbor index wrapper for item embeddings.

Spotlight stores the full item-embedding matrix (50K+ vectors) in a FAISS index
so that, at serve time, a single user embedding can be matched against the entire
catalog in sub-millisecond time via approximate nearest-neighbor (ANN) search.

Two index types are supported:

* ``IndexFlatIP``  — exact, brute-force inner-product search. O(N*d) per query.
  Perfect recall; used as a correctness baseline and fine up to a few hundred K.
* ``IndexIVFFlat`` — inverted-file (coarse-quantizer) index. Vectors are
  clustered into ``nlist`` Voronoi cells; a query only scans the ``nprobe``
  nearest cells, trading a little recall for a large speedup. This is the
  production path for large catalogs.

Because the towers emit L2-normalized embeddings, inner product == cosine
similarity, so the IP metric returns cosine-nearest items.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised only without faiss
    raise ImportError(
        "FAISS is required for Spotlight's ANN index but is not installed.\n"
        "Install it with:  pip install faiss-cpu\n"
        "(or faiss-gpu if you have CUDA). See requirements.txt."
    ) from exc


class ItemIndex:
    """
    Wraps a FAISS index over item embeddings plus the item-id mapping.

    The FAISS index works on contiguous row positions (0..N-1); ``item_ids`` maps
    those positions back to the original catalog ids so queries return real ids.

    Args:
        index: a built FAISS index.
        item_ids: (N,) int64 mapping from FAISS row -> catalog item id.
        dim: embedding dimensionality.
        index_type: "flat" or "ivf" (recorded for save/load metadata).
        nprobe: query-time cells to visit (IVF only).
    """

    def __init__(
        self,
        index: "faiss.Index",
        item_ids: np.ndarray,
        dim: int,
        index_type: str,
        nprobe: int = 16,
    ):
        self.index = index
        self.item_ids = item_ids.astype(np.int64)
        self.dim = dim
        self.index_type = index_type
        self.nprobe = nprobe
        if index_type == "ivf":
            # nprobe controls the recall/latency tradeoff at query time.
            self.index.nprobe = nprobe

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
        index_type: str = "ivf",
        nlist: int = 256,
        nprobe: int = 16,
        metric: str = "ip",
        seed: int = 42,
    ) -> "ItemIndex":
        """
        Build an index from a matrix of item embeddings.

        Args:
            embeddings: (N, d) float32 item embeddings (should be L2-normalized).
            item_ids: (N,) catalog item ids aligned with ``embeddings`` rows.
            index_type: "flat" (exact) or "ivf" (approximate).
            nlist: number of IVF Voronoi cells (ignored for flat).
            nprobe: IVF query-time cells to scan (ignored for flat).
            metric: "ip" for inner product (cosine on unit vectors) or "l2".
            seed: FAISS clustering seed for reproducible IVF training.

        Returns:
            A ready-to-query ItemIndex.
        """
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        n, dim = embeddings.shape

        if metric == "ip":
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric {metric!r}; use 'ip' or 'l2'.")

        if index_type == "flat":
            if faiss_metric == faiss.METRIC_INNER_PRODUCT:
                index: "faiss.Index" = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
        elif index_type == "ivf":
            # Clamp nlist so each cell has a sane number of points.
            effective_nlist = max(1, min(nlist, n // 39 if n >= 39 else 1))
            quantizer = (
                faiss.IndexFlatIP(dim)
                if faiss_metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss_metric)
            faiss.cvar.rand_seed = seed  # reproducible k-means init
            # IVF must be trained on the data to learn the Voronoi partition.
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type {index_type!r}; use 'flat' or 'ivf'.")

        index.add(embeddings)
        return cls(
            index=index,
            item_ids=item_ids,
            dim=dim,
            index_type=index_type,
            nprobe=nprobe,
        )

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(
        self, queries: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the top-k items for each query embedding.

        Args:
            queries: (Q, d) float32 query (user) embeddings.
            k: number of neighbors to return per query.

        Returns:
            scores: (Q, k) similarity scores (inner product / cosine).
            ids: (Q, k) catalog item ids (mapped through ``item_ids``).
        """
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if queries.ndim == 1:
            queries = queries[None, :]
        scores, positions = self.index.search(queries, k)
        # Map FAISS row positions back to catalog item ids. -1 marks "not found"
        # (can happen for IVF when fewer than k candidates are in scanned cells).
        ids = np.where(positions >= 0, self.item_ids[positions.clip(min=0)], -1)
        return scores, ids

    @property
    def ntotal(self) -> int:
        """Number of vectors stored in the index."""
        return int(self.index.ntotal)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """
        Persist the FAISS index plus its sidecar id-mapping/metadata.

        Writes ``<path>`` (the binary FAISS index) and ``<path>.meta.json``
        (item ids + index parameters).

        Args:
            path: output path for the FAISS index (e.g. ``index/artifacts/items.faiss``).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        faiss.write_index(self.index, path)
        meta = {
            "item_ids": self.item_ids.tolist(),
            "dim": self.dim,
            "index_type": self.index_type,
            "nprobe": self.nprobe,
        }
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh)
        print(f"FAISS index ({self.ntotal} vectors) saved to {path}")

    @classmethod
    def load(cls, path: str, nprobe: Optional[int] = None) -> "ItemIndex":
        """
        Load an index previously written by :meth:`save`.

        Args:
            path: path to the FAISS index file.
            nprobe: optional override for IVF query-time cells.

        Returns:
            A ready-to-query ItemIndex.
        """
        index = faiss.read_index(path)
        with open(path + ".meta.json", "r") as fh:
            meta = json.load(fh)
        return cls(
            index=index,
            item_ids=np.asarray(meta["item_ids"], dtype=np.int64),
            dim=int(meta["dim"]),
            index_type=meta["index_type"],
            nprobe=nprobe if nprobe is not None else int(meta["nprobe"]),
        )
