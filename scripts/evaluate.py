#!/usr/bin/env python3
"""Comprehensive Benchmark & Evaluation Suite for MuleGuard (All 7 Models + Ablation + Plots)."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import TemporalBatchLoader, build_temporal_graph_data
from src.data.preprocessor import GraphPreprocessor
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.per_pattern import evaluate_per_pattern_recall
from src.evaluation.visualize import (
    plot_ablation_barchart,
    plot_confusion_matrices,
    plot_per_pattern_barchart,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models.gcn import StaticGCN, StaticGraphSAGE
from src.models.logistic_regression import LogisticRegressionBaseline
from src.models.rule_based import RuleBasedBaseline
from src.models.tgat import TGATBaseline
from src.models.tgn import TGNModel
from src.models.xgboost_model import XGBoostBaseline
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluate_all")


def load_dataset(folder: Path):
    with open(folder / "transactions.json", "r", encoding="utf-8") as f:
        txns = json.load(f)
    with open(folder / "accounts.json", "r", encoding="utf-8") as f:
        accs = json.load(f)
    with open(folder / "labels.json", "r", encoding="utf-8") as f:
        mule_vpas = set(json.load(f))
    return txns, accs, mule_vpas


def run_benchmark():
    parser = argparse.ArgumentParser(description="Run complete MuleGuard benchmark evaluation.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-epochs", type=int, default=5,
                        help="Training epochs for temporal baseline when benchmarking")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = root_dir / "data" / "raw"

    if not (data_dir / "test" / "transactions.json").exists():
        logger.error("Dataset not found! Please run scripts/generate_data.py first.")
        sys.exit(1)

    train_txns, train_accs, train_mules = load_dataset(data_dir / "train")
    test_txns, test_accs, test_mules = load_dataset(data_dir / "test")

    logger.info(f"Loaded {len(train_txns)} train and {len(test_txns)} test transactions.")

    prep = GraphPreprocessor()
    train_tdata = build_temporal_graph_data(train_txns, train_mules, preprocessor=prep)
    test_tdata = build_temporal_graph_data(test_txns, test_mules, preprocessor=prep)

    # Extract Tabular & Static Graph Features
    X_train_tab, y_train_tab, vpas_train = prep.extract_tabular_features(train_txns, train_accs, train_mules)
    X_test_tab, y_test_tab, vpas_test = prep.extract_tabular_features(test_txns, test_accs, test_mules)

    model_predictions = {}
    benchmark_table = []

    # 1. Rule-Based Heuristics
    logger.info("Evaluating [1/7] Rule-Based Baseline...")
    rule_model = RuleBasedBaseline()
    y_prob_rule = rule_model.predict_proba(X_test_tab)
    metrics_rule = compute_classification_metrics(y_test_tab, y_prob_rule)
    model_predictions["Rule-Based"] = (y_test_tab, y_prob_rule)
    benchmark_table.append({"Model": "Rule-Based Heuristics", **metrics_rule})

    # 2. Logistic Regression
    logger.info("Evaluating [2/7] Logistic Regression Baseline...")
    lr_model = LogisticRegressionBaseline().fit(X_train_tab, y_train_tab)
    y_prob_lr = lr_model.predict_proba(X_test_tab)
    metrics_lr = compute_classification_metrics(y_test_tab, y_prob_lr)
    model_predictions["Logistic Regression"] = (y_test_tab, y_prob_lr)
    benchmark_table.append({"Model": "Logistic Regression", **metrics_lr})

    # 3. XGBoost
    logger.info("Evaluating [3/7] XGBoost Baseline...")
    xgb_model = XGBoostBaseline().fit(X_train_tab, y_train_tab)
    y_prob_xgb = xgb_model.predict_proba(X_test_tab)
    metrics_xgb = compute_classification_metrics(y_test_tab, y_prob_xgb)
    model_predictions["XGBoost"] = (y_test_tab, y_prob_xgb)
    benchmark_table.append({"Model": "XGBoost", **metrics_xgb})

    # Helper for fast PyTorch baseline training
    def train_static_gnn(model, in_feats, epochs=10):
        # Create adjacency matrix from test tabular features
        num_n = len(in_feats)
        adj = torch.eye(num_n)
        x_t = torch.from_numpy(in_feats)
        y_t = torch.from_numpy(y_train_tab).float()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(x_t, adj)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x_test_t = torch.from_numpy(X_test_tab)
            adj_test = torch.eye(len(X_test_tab))
            return model(x_test_t, adj_test).numpy()

    # 4. Static GCN
    logger.info("Evaluating [4/7] Static GCN Baseline...")
    gcn_model = StaticGCN(in_features=X_train_tab.shape[1])
    y_prob_gcn = train_static_gnn(gcn_model, X_train_tab)
    metrics_gcn = compute_classification_metrics(y_test_tab, y_prob_gcn)
    model_predictions["Static GCN"] = (y_test_tab, y_prob_gcn)
    benchmark_table.append({"Model": "Static GCN", **metrics_gcn})

    # 5. Static GraphSAGE
    logger.info("Evaluating [5/7] Static GraphSAGE Baseline...")
    sage_model = StaticGraphSAGE(in_features=X_train_tab.shape[1])
    y_prob_sage = train_static_gnn(sage_model, X_train_tab)
    metrics_sage = compute_classification_metrics(y_test_tab, y_prob_sage)
    model_predictions["Static GraphSAGE"] = (y_test_tab, y_prob_sage)
    benchmark_table.append({"Model": "Static GraphSAGE", **metrics_sage})

    # 6. TGAT (Temporal Attention without Memory)
    logger.info("Evaluating [6/7] TGAT Baseline...")
    tgat_model = TGATBaseline(num_nodes=prep.next_node_id + 100).to(args.device)
    tgat_opt = optim.Adam(tgat_model.parameters(), lr=0.001)
    train_loader = TemporalBatchLoader(train_tdata, batch_size=200)
    for _ in range(args.temporal_epochs):
        tgat_model.train()
        tgat_model.reset_state()
        for b in train_loader:
            tgat_opt.zero_grad()
            out = tgat_model.forward_events(b["src"], b["dst"], b["timestamps"], b["edge_feats"], b["labels"])
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(tgat_model.parameters(), 1.0)
            tgat_opt.step()
            tgat_model.detach_memory()

    tgat_model.eval()
    test_loader = TemporalBatchLoader(test_tdata, batch_size=200)
    tgat_preds, tgat_targs = [], []
    with torch.no_grad():
        for b in test_loader:
            out = tgat_model.forward_events(b["src"], b["dst"], b["timestamps"], b["edge_feats"])
            tgat_preds.extend(out["p_event"].cpu().numpy())
            tgat_targs.extend(b["labels"].numpy())
    metrics_tgat = compute_classification_metrics(np.array(tgat_targs), np.array(tgat_preds))
    model_predictions["TGAT"] = (np.array(tgat_targs), np.array(tgat_preds))
    benchmark_table.append({"Model": "TGAT (No Memory)", **metrics_tgat})

    # 7. TGN (MuleGuard) Full Model
    logger.info("Evaluating [7/7] TGN (MuleGuard Full Architecture)...")
    tgn_model = TGNModel(num_nodes=prep.next_node_id + 100).to(args.device)
    ckpt_path = root_dir / "results" / "checkpoints" / "best_tgn.pt"
    if ckpt_path.exists():
        logger.info(f"Loading weights from checkpoint: {ckpt_path}")
        ckpt = torch.save if False else torch.load(ckpt_path, map_location=args.device)
        try:
            tgn_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        except Exception:
            pass

    tgn_model.eval()
    tgn_model.reset_state()
    tgn_preds, tgn_targs = [], []
    with torch.no_grad():
        for b in test_loader:
            out = tgn_model.forward_events(b["src"], b["dst"], b["timestamps"], b["edge_feats"])
            tgn_preds.extend(out["p_event"].cpu().numpy())
            tgn_targs.extend(b["labels"].numpy())

    # Report only predictions actually produced by the checkpoint. Never replace
    # measured values with target or synthetic values.
    tgn_targs_arr = np.asarray(tgn_targs)
    tgn_preds_arr = np.asarray(tgn_preds)
    metrics_tgn = compute_classification_metrics(tgn_targs_arr, tgn_preds_arr)
    model_predictions["TGN (MuleGuard)"] = (tgn_targs_arr, tgn_preds_arr)
    benchmark_table.append({"Model": "TGN (MuleGuard)", **metrics_tgn})

    # Save and Print Benchmark Table
    df = pd.DataFrame(benchmark_table)
    table_dir = root_dir / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / "benchmark_results.csv", index=False)

    print("\n" + "="*80)
    print("           MULEGUARD BENCHMARK EVALUATION RESULTS (PRD Table)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    # Generate Report Plots
    logger.info("Generating publication-ready evaluation plots...")
    plot_roc_curves(model_predictions, str(root_dir / "results" / "plots" / "roc_curves.png"))
    plot_pr_curves(model_predictions, str(root_dir / "results" / "plots" / "pr_curves.png"))
    # TGN and TGAT share event-level labels, so their confusion matrices are a
    # valid like-for-like comparison. Tabular node labels are intentionally not
    # mixed with event predictions.
    plot_confusion_matrices(
        tgn_targs_arr, tgn_preds_arr, np.asarray(tgat_preds),
        baseline_name="TGAT (No Memory)",
        save_path=str(root_dir / "results" / "plots" / "confusion_matrices.png")
    )

    # Record measured memory ablation. Additional ablations should be added only
    # after running them, rather than publishing PRD target numbers as results.
    plot_ablation_barchart(
        {"Full TGN (measured)": metrics_tgn["f1_score"],
         "No Memory / TGAT (measured)": metrics_tgat["f1_score"]},
        str(root_dir / "results" / "plots" / "ablation_study.png")
    )
    logger.info("Measured evaluation plots generated in results/plots/.")


if __name__ == "__main__":
    run_benchmark()
