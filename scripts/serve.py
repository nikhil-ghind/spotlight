"""
CLI: serving path. Given a user id, embed with the user tower and ANN-search the
FAISS item index for the top-k recommendations, printing sub-millisecond timing.

Run from the ``spotlight/`` root:

    python -m scripts.serve --config configs/config.yaml \
        --data data/artifacts/synthetic.npz \
        --checkpoint checkpoints/two_tower_best.pt \
        --user-id 1234 --k 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from data import synthetic
from index.faiss_index import ItemIndex
from models.two_tower import TwoTowerModel
from scripts._common import embed_users, load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve top-k recommendations.")
    parser.add_argument("--config", default="configs/config.yaml", help="config path")
    parser.add_argument(
        "--data",
        default="data/artifacts/synthetic.npz",
        help="path to the synthetic .npz dataset",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/two_tower_best.pt",
        help="trained two-tower checkpoint",
    )
    parser.add_argument(
        "--index",
        default="index/artifacts/items.faiss",
        help="path to the FAISS index",
    )
    parser.add_argument("--user-id", type=int, default=0, help="user id to query")
    parser.add_argument("--k", type=int, default=10, help="number of items to return")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="repeat the ANN search N times to report a stable mean latency",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]

    device = pick_device(args.device)
    data = synthetic.load(args.data)
    if not (0 <= args.user_id < data.n_users):
        raise SystemExit(f"user-id must be in [0, {data.n_users}); got {args.user_id}")

    model = TwoTowerModel.load(
        args.checkpoint,
        device=str(device),
        feature_embedding_dim=mcfg["feature_embedding_dim"],
        tower_hidden=tuple(mcfg["tower_hidden"]),
        dropout=mcfg["dropout"],
        l2_normalize=mcfg["l2_normalize"],
    )
    index = ItemIndex.load(args.index, nprobe=cfg["index"]["nprobe"])

    # Embed the requested user once with the user tower.
    user_emb = embed_users(
        model, data, np.asarray([args.user_id], dtype=np.int64), device
    )

    # Warm up (first FAISS call pays one-time allocation costs).
    index.search(user_emb, k=args.k)

    # Time the ANN search across N trials for a stable sub-ms estimate.
    t0 = time.perf_counter()
    for _ in range(args.n_trials):
        scores, ids = index.search(user_emb, k=args.k)
    elapsed = time.perf_counter() - t0
    per_query_ms = (elapsed / args.n_trials) * 1000.0

    print(f"User {args.user_id} | index={index.ntotal} items | top-{args.k}:")
    for rank, (item_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
        print(f"  {rank:2d}. item {int(item_id):6d}   cos_sim={score:.4f}")
    print(
        f"\nANN search latency: {per_query_ms:.3f} ms/query "
        f"(mean over {args.n_trials} trials, nprobe={index.nprobe})"
    )


if __name__ == "__main__":
    main()
