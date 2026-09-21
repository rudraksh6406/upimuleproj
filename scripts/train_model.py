#!/usr/bin/env python3
"""Training script for Temporal Graph Neural Network (TGN)."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import TemporalBatchLoader, build_temporal_graph_data
from src.data.preprocessor import GraphPreprocessor
from src.evaluation.metrics import compute_classification_metrics
from src.models.tgn import TGNModel
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_tgn")


def load_dataset_from_disk(folder: Path):
    with open(folder / "transactions.json", "r", encoding="utf-8") as f:
        txns = json.load(f)
    with open(folder / "labels.json", "r", encoding="utf-8") as f:
        mule_vpas = set(json.load(f))
    return txns, mule_vpas


def train():
    parser = argparse.ArgumentParser(description="Train MuleGuard TGN Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda or mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config("model_config.yaml")
    m_cfg = cfg.get("model", {})
    
    device = "mps" if (torch.backends.mps.is_available() and args.device == "mps") else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Load Data
    data_dir = root_dir / "data" / "raw"
    if not (data_dir / "train" / "transactions.json").exists():
        logger.error("Dataset not found! Please run scripts/generate_data.py first.")
        sys.exit(1)

    train_txns, train_mules = load_dataset_from_disk(data_dir / "train")
    val_txns, val_mules = load_dataset_from_disk(data_dir / "val")

    logger.info(f"Loaded {len(train_txns)} train txns and {len(val_txns)} val txns.")

    # Preprocess into Temporal Graph representations
    prep = GraphPreprocessor()
    train_data = build_temporal_graph_data(train_txns, train_mules, preprocessor=prep)
    val_data = build_temporal_graph_data(val_txns, val_mules, preprocessor=prep)

    logger.info(f"Graph initialized with {prep.next_node_id} unique accounts.")

    # Initialize TGN Model
    model = TGNModel(
        num_nodes=prep.next_node_id + 500,  # headroom for new streaming nodes
        memory_dim=m_cfg.get("memory_dim", 64),
        embedding_dim=m_cfg.get("embedding_dim", 128),
        message_dim=m_cfg.get("message_dim", 64),
        edge_feat_dim=8,
        time_dim=32,
        attention_heads=m_cfg.get("attention_heads", 4),
        dropout=m_cfg.get("dropout", 0.2),
        neighborhood_size=m_cfg.get("neighborhood_size", 20),
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = 0.0
    checkpoint_dir = root_dir / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_tgn.pt"

    train_loader = TemporalBatchLoader(train_data, batch_size=args.batch_size)
    val_loader = TemporalBatchLoader(val_data, batch_size=args.batch_size)

    logger.info("Starting TGN training loop...")

    for epoch in range(1, args.epochs + 1):
        # --- TRAIN ---
        model.train()
        model.reset_state()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass over temporal batch
            out = model.forward_events(
                src=batch["src"],
                dst=batch["dst"],
                timestamps=batch["timestamps"],
                edge_feats=batch["edge_feats"],
                labels=batch["labels"],
            )

            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Detach memory for truncated BPTT across batches
            model.detach_memory()

            total_loss += loss.item()
            train_preds.extend(out["p_event"].detach().cpu().numpy())
            train_targets.extend(batch["labels"].numpy())

        train_metrics = compute_classification_metrics(np.array(train_targets), np.array(train_preds))

        # --- VALIDATE ---
        model.eval()
        model.reset_state()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                out = model.forward_events(
                    src=batch["src"],
                    dst=batch["dst"],
                    timestamps=batch["timestamps"],
                    edge_feats=batch["edge_feats"],
                    labels=batch["labels"],
                )
                val_preds.extend(out["p_event"].cpu().numpy())
                val_targets.extend(batch["labels"].numpy())

        val_metrics = compute_classification_metrics(np.array(val_targets), np.array(val_preds))
        scheduler.step(val_metrics["f1_score"])

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Train F1: {train_metrics['f1_score']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
        )

        # Save Best Checkpoint
        if val_metrics["f1_score"] >= best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "node_to_idx": prep.node_to_idx,
                "val_f1": best_val_f1,
                "val_auc": val_metrics["roc_auc"],
            }, best_model_path)
            logger.info(f"==> Saved new best model checkpoint to {best_model_path}")

    logger.info(f"Training completed! Best Validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()
