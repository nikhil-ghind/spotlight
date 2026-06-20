"""
CLI: train the two-tower model with in-batch negative sampling.

Run from the ``spotlight/`` root:

    python -m scripts.train --config configs/config.yaml \
        --data data/artifacts/synthetic.npz

Logs train loss / in-batch recall@1 periodically and saves the best checkpoint
(by validation loss) to ``train.out_dir``.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import synthetic
from data.dataset import InteractionPairDataset, collate_pairs
from models.two_tower import TwoTowerModel
from scripts._common import load_config, pick_device


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move every tensor in a batch dict onto ``device``."""
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: TwoTowerModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Run the in-batch contrastive objective over a loader (no grad)."""
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        _, metrics = model.compute_loss(batch)
        total_loss += metrics["loss"]
        total_recall += metrics["recall_at_1"]
        n_batches += 1
    n_batches = max(n_batches, 1)
    return {
        "val_loss": total_loss / n_batches,
        "val_recall@1": total_recall / n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Spotlight two-tower model.")
    parser.add_argument("--config", default="configs/config.yaml", help="config path")
    parser.add_argument(
        "--data",
        default="data/artifacts/synthetic.npz",
        help="path to the synthetic .npz dataset",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    mcfg = cfg["model"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]

    # Reproducibility.
    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    device = pick_device(args.device)
    print(f"Loading dataset from {args.data} ...")
    data = synthetic.load(args.data)
    print(
        f"Dataset: {data.n_users} users, {data.n_items} items, "
        f"{len(data.train_pairs)} train pairs, {len(data.val_pairs)} val pairs"
    )

    train_ds = InteractionPairDataset(data, split="train")
    val_ds = InteractionPairDataset(data, split="val")
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        collate_fn=collate_pairs,
        drop_last=True,  # keep the in-batch negative count constant
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        collate_fn=collate_pairs,
    )

    model = TwoTowerModel(
        spec=data.spec,
        embedding_dim=mcfg["embedding_dim"],
        feature_embedding_dim=mcfg["feature_embedding_dim"],
        tower_hidden=tuple(mcfg["tower_hidden"]),
        dropout=mcfg["dropout"],
        temperature=tcfg["temperature"],
        l2_normalize=mcfg["l2_normalize"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    out_dir = tcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "two_tower_best.pt")
    best_val = float("inf")

    print(f"Training on {device} for {epochs} epoch(s) | batch_size={tcfg['batch_size']} "
          f"(=> {tcfg['batch_size'] - 1} in-batch negatives/example)")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_recall = 0.0
        seen = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            loss, metrics = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()

            global_step += 1
            run_loss += metrics["loss"]
            run_recall += metrics["recall_at_1"]
            seen += 1

            if global_step % tcfg["log_every"] == 0:
                print(
                    f"  step {global_step:6d} | loss={run_loss / seen:.4f} "
                    f"| ib_recall@1={run_recall / seen:.3f} "
                    f"| pos_sim={metrics['pos_sim']:.3f} neg_sim={metrics['neg_sim']:.3f}"
                )
                run_loss = 0.0
                run_recall = 0.0
                seen = 0

        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d} | val_loss={val_metrics['val_loss']:.4f} "
            f"| val_ib_recall@1={val_metrics['val_recall@1']:.3f} | time={dt:.1f}s"
        )

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            model.save(best_path, data.spec)

    print(f"Done. Best val_loss={best_val:.4f} | checkpoint: {best_path}")


if __name__ == "__main__":
    main()
