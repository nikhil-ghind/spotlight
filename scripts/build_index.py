"""
CLI: the "nightly refresh" batch job.

Loads the trained item tower, embeds the *entire* item catalog (50K+ items),
builds a FAISS ANN index over those embeddings, and persists it (plus the id
mapping) so the serving path can do sub-millisecond retrieval.

Run from the ``spotlight/`` root:

    python -m scripts.build_index --config configs/config.yaml \
        --data data/artifacts/synthetic.npz \
        --checkpoint checkpoints/two_tower_best.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from data import synthetic
from index.faiss_index import ItemIndex
from models.two_tower import TwoTowerModel
from scripts._common import embed_all_items, load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the nightly FAISS item index.")
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
        "--out",
        default=None,
        help="output .faiss path (defaults to <index.out_dir>/items.faiss)",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    icfg = cfg["index"]
    mcfg = cfg["model"]

    device = pick_device(args.device)
    print(f"Loading dataset from {args.data} ...")
    data = synthetic.load(args.data)

    print(f"Loading two-tower checkpoint from {args.checkpoint} ...")
    model = TwoTowerModel.load(
        args.checkpoint,
        device=str(device),
        feature_embedding_dim=mcfg["feature_embedding_dim"],
        tower_hidden=tuple(mcfg["tower_hidden"]),
        dropout=mcfg["dropout"],
        l2_normalize=mcfg["l2_normalize"],
    )

    print(f"Embedding all {data.n_items} items with the item tower ...")
    t0 = time.time()
    embeddings = embed_all_items(model, data, device)
    embed_dt = time.time() - t0
    item_ids = np.arange(data.n_items, dtype=np.int64)
    print(f"  embedded {data.n_items} items in {embed_dt:.1f}s "
          f"(dim={embeddings.shape[1]})")

    print(f"Building FAISS index (type={icfg['type']}, metric={icfg['metric']}) ...")
    t0 = time.time()
    index = ItemIndex.build(
        embeddings=embeddings,
        item_ids=item_ids,
        index_type=icfg["type"],
        nlist=icfg["nlist"],
        nprobe=icfg["nprobe"],
        metric=icfg["metric"],
        seed=cfg["train"]["seed"],
    )
    build_dt = time.time() - t0

    out_dir = icfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "items.faiss")
    index.save(out_path)
    print(f"Nightly refresh complete | build={build_dt:.1f}s | index={out_path}")


if __name__ == "__main__":
    main()
