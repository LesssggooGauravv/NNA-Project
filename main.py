"""
main.py — Full Experiment Orchestrator

GRU vs LSTM Under Memory Stress
================================
This script runs the complete benchmark:
  1. Generates synthetic long-sequence datasets for each sequence length
  2. Trains RNN, GRU, LSTM, and Attn-LSTM on every length
  3. Records training loss, validation accuracy, and gradient norms
  4. Saves CSV logs to results/
  5. Generates publication-quality comparison plots to plots/

Run:
    py main.py                     # default config
    py main.py --epochs 50         # override epochs
    py main.py --quick             # fast sanity-check mode (fewer samples, shorter seqs)

Author : GRU-vs-LSTM Research Project
"""

import argparse
import os
import time
import json
import torch

from src.dataset import build_dataloaders
from src.models import build_model, MODEL_REGISTRY
from src.trainer import run_training
from src.metrics import MetricsLogger
from src.visualization import generate_all_plots


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "sequence_lengths": [10, 50, 100, 200, 500],
    "models": ["RNN", "GRU", "LSTM", "Attn-LSTM"],
    "num_samples": 10_000,
    "num_classes": 8,
    "noise_std": 1.0,
    "batch_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.0,
    "epochs": 30,
    "learning_rate": 1e-3,
    "clip_value": 5.0,
    "train_ratio": 0.8,
    "seed": 42,
    "results_dir": "results",
    "plots_dir": "plots",
}

QUICK_CONFIG = {
    **DEFAULT_CONFIG,
    "sequence_lengths": [10, 50, 100],
    "num_samples": 2_000,
    "epochs": 10,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRU vs LSTM Memory Stress Benchmark"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run a fast sanity check with reduced params")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None,
                        help="Override sequence lengths to test")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Override which models to train")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = QUICK_CONFIG.copy() if args.quick else DEFAULT_CONFIG.copy()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seq_lengths is not None:
        cfg["sequence_lengths"] = args.seq_lengths
    if args.models is not None:
        cfg["models"] = args.models
    if args.hidden_size is not None:
        cfg["hidden_size"] = args.hidden_size
    if args.num_samples is not None:
        cfg["num_samples"] = args.num_samples
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  GRU vs LSTM — Memory Stress Benchmark")
    print(f"{'='*65}")
    print(f"  Device         : {device}")
    print(f"  Models         : {cfg['models']}")
    print(f"  Seq Lengths    : {cfg['sequence_lengths']}")
    print(f"  Samples        : {cfg['num_samples']}")
    print(f"  Epochs         : {cfg['epochs']}")
    print(f"  Hidden Size    : {cfg['hidden_size']}")
    print(f"  Learning Rate  : {cfg['learning_rate']}")
    print(f"{'='*65}\n")

    # Save config for reproducibility
    os.makedirs(cfg["results_dir"], exist_ok=True)
    os.makedirs(cfg["plots_dir"], exist_ok=True)
    with open(os.path.join(cfg["results_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Storage ──────────────────────────────────────────────────────
    # results[model_name][seq_len] = MetricsLogger
    results: dict[str, dict[int, MetricsLogger]] = {}
    # times[model_name][seq_len] = float (seconds)
    train_times: dict[str, dict[int, float]] = {}

    total_runs = len(cfg["models"]) * len(cfg["sequence_lengths"])
    run_idx = 0

    for model_name in cfg["models"]:
        results[model_name] = {}
        train_times[model_name] = {}

        for seq_len in cfg["sequence_lengths"]:
            run_idx += 1
            print(f"\n{'─'*55}")
            print(f"  [{run_idx}/{total_runs}]  {model_name}  |  seq_len={seq_len}")
            print(f"{'─'*55}")

            # 1. Data
            train_loader, test_loader = build_dataloaders(
                seq_len=seq_len,
                batch_size=cfg["batch_size"],
                num_samples=cfg["num_samples"],
                num_classes=cfg["num_classes"],
                noise_std=cfg["noise_std"],
                train_ratio=cfg["train_ratio"],
                seed=cfg["seed"],
            )

            # 2. Model
            model = build_model(
                name=model_name,
                input_size=1,
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                num_classes=cfg["num_classes"],
                dropout=cfg["dropout"],
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")

            # 3. Train
            t0 = time.time()
            logger = run_training(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=cfg["epochs"],
                lr=cfg["learning_rate"],
                clip_value=cfg["clip_value"],
                device=device,
                verbose=True,
            )
            elapsed = time.time() - t0

            # 4. Store results
            results[model_name][seq_len] = logger
            train_times[model_name][seq_len] = elapsed

            # 5. Save CSV
            csv_path = os.path.join(
                cfg["results_dir"],
                f"{model_name.replace('-','_')}_seq{seq_len}.csv",
            )
            logger.save_csv(csv_path)

            best_acc = logger.get_best_val_acc() * 100
            print(f"  ✅ Best Val Acc: {best_acc:.2f}%  |  Time: {elapsed:.1f}s")

    # ── Summary Table ────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  📊  RESULTS SUMMARY — Best Validation Accuracy (%)")
    print(f"{'='*65}")
    header = f"{'Model':<12}" + "".join(f"{'SL='+str(sl):>10}" for sl in cfg["sequence_lengths"])
    print(f"  {header}")
    print(f"  {'─'*len(header)}")
    for mn in cfg["models"]:
        row = f"  {mn:<12}"
        for sl in cfg["sequence_lengths"]:
            acc = results[mn][sl].get_best_val_acc() * 100
            row += f"{acc:>10.2f}"
        print(row)
    print()

    # ── Time Table ───────────────────────────────────────────────────
    print(f"  ⏱️  Training Time (seconds)")
    print(f"  {'─'*len(header)}")
    for mn in cfg["models"]:
        row = f"  {mn:<12}"
        for sl in cfg["sequence_lengths"]:
            t = train_times[mn][sl]
            row += f"{t:>10.1f}"
        print(row)
    print()

    # ── Generate plots ───────────────────────────────────────────────
    generate_all_plots(results, train_times, cfg["sequence_lengths"], cfg["plots_dir"])

    print(f"\n{'='*65}")
    print(f"  ✅  Experiment complete!")
    print(f"  📂  Results → {cfg['results_dir']}/")
    print(f"  📊  Plots   → {cfg['plots_dir']}/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
