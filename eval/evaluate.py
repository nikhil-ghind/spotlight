"""
CLI: evaluate retrieval quality on held-out positives using the FAISS index.

For each held-out (user, item*) pair we embed the user, retrieve the top
max(ks) items from the index, and score Recall@K and MRR against item*.

Run from the ``spotlight/`` root:

    python -m eval.evaluate --config configs/config.yaml \
        --data data/artifacts/synthetic.npz \
        --checkpoint checkpoints/two_tower_best.pt \
        --index index/artifacts/items.faiss
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from data import synthetic
from eval.metrics import evaluate_retrieval
from index.faiss_index import ItemIndex
from models.two_tower import TwoTowerModel
from scripts._common import embed_users, load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Spotlight retrieval.")
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
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ks = list(cfg["eval"]["ks"])
    k_max = max(ks)

    device = pick_device(args.device)
    data = synthetic.load(args.data)
    val_pairs = data.val_pairs
    if len(val_pairs) == 0:
        raise SystemExit("No validation pairs found; regenerate data with val_frac > 0.")

    model = TwoTowerModel.load(
        args.checkpoint,
        device=str(device),
        feature_embedding_dim=mcfg["feature_embedding_dim"],
        tower_hidden=tuple(mcfg["tower_hidden"]),
        dropout=mcfg["dropout"],
        l2_normalize=mcfg["l2_normalize"],
    )
    index = ItemIndex.load(args.index, nprobe=cfg["index"]["nprobe"])

    val_user_ids = val_pairs[:, 0]
    val_item_ids = val_pairs[:, 1]

    print(f"Embedding {len(val_user_ids)} held-out users ...")
    user_emb = embed_users(model, data, val_user_ids, device)

    print(f"Retrieving top-{k_max} for each user from a {index.ntotal}-item index ...")
    t0 = time.perf_counter()
    _, retrieved_ids = index.search(user_emb, k=k_max)
    dt = time.perf_counter() - t0
    per_query_ms = (dt / max(len(val_user_ids), 1)) * 1000.0

    results = evaluate_retrieval(retrieved_ids, val_item_ids, ks=ks)

    print("\n=== Retrieval evaluation (held-out positives) ===")
    for k in ks:
        print(f"  Recall@{k:<3d} : {results[f'recall@{k}']:.4f}")
    print(f"  MRR        : {results['mrr']:.4f}")
    print(
        f"\nBatched ANN throughput: {per_query_ms:.4f} ms/query "
        f"over {len(val_user_ids)} queries (nprobe={index.nprobe})"
    )


if __name__ == "__main__":
    main()
