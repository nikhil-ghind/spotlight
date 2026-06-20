"""
CLI: generate synthetic interaction logs and write them to disk.

Run from the ``spotlight/`` root:

    python -m scripts.generate_data --config configs/config.yaml

Writes a compressed ``.npz`` archive (user/item features + train/val pairs) to
``data.out_dir`` from the config.
"""

from __future__ import annotations

import argparse
import os
import time

from data import synthetic
from scripts._common import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Spotlight synthetic data.")
    parser.add_argument("--config", default="configs/config.yaml", help="config path")
    parser.add_argument(
        "--out",
        default=None,
        help="output .npz path (defaults to <data.out_dir>/synthetic.npz)",
    )
    parser.add_argument(
        "--n-items", type=int, default=None, help="override item catalog size"
    )
    parser.add_argument(
        "--n-users", type=int, default=None, help="override number of users"
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=None,
        help="override number of interaction logs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]

    n_users = args.n_users if args.n_users is not None else d["n_users"]
    n_items = args.n_items if args.n_items is not None else d["n_items"]
    n_interactions = (
        args.n_interactions
        if args.n_interactions is not None
        else d["n_interactions"]
    )

    out_dir = d["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "synthetic.npz")

    print(
        f"Generating synthetic data: {n_users} users, {n_items} items, "
        f"{n_interactions} interactions (seed={cfg['train']['seed']})..."
    )
    t0 = time.time()
    dataset = synthetic.generate(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions,
        n_latent=d["n_latent"],
        user_categorical_cardinalities=(
            d["user_n_countries"],
            d["user_n_age_buckets"],
        ),
        item_categorical_cardinalities=(
            d["item_n_categories"],
            d["item_n_brands"],
        ),
        user_n_numeric=d["user_n_numeric"],
        item_n_numeric=d["item_n_numeric"],
        val_frac=d["val_frac"],
        seed=cfg["train"]["seed"],
    )
    synthetic.save(dataset, out_path)
    dt = time.time() - t0

    print(
        f"Wrote {out_path}  | train_pairs={len(dataset.train_pairs)} "
        f"val_pairs={len(dataset.val_pairs)} | {dt:.1f}s"
    )


if __name__ == "__main__":
    main()
