"""
metrics.py — Evaluation Metrics & Gradient Monitoring Utilities

Provides:
  • Accuracy computation
  • Per-layer gradient L2-norm extraction (the core of the gradient
    analysis that demonstrates vanishing / exploding gradients)
  • A lightweight MetricsLogger that accumulates per-epoch stats and
    serialises them to CSV for reproducibility.

Author : GRU-vs-LSTM Research Project
"""

import torch
import torch.nn as nn
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
#  Accuracy
# ──────────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return accuracy as a float in [0, 1]."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ──────────────────────────────────────────────────────────────────────
#  Gradient Norm Extraction
# ──────────────────────────────────────────────────────────────────────

def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    After a backward() call, iterate over the model's named parameters
    and collect the L2 norm of each gradient tensor.

    Returns a dict like:
        {"rnn.weight_hh_l0": 0.034, "rnn.weight_ih_l1": 1.23, ...}

    Only recurrent weights (weight_hh, weight_ih) are included because
    they are the ones affected by vanishing/exploding gradients through
    BPTT unrolling.
    """
    norms: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None and ("weight_hh" in name or "weight_ih" in name):
            norms[name] = param.grad.data.norm(2).item()
    return norms


def compute_total_gradient_norm(model: nn.Module) -> float:
    """Return the total L2 norm across all recurrent weight gradients."""
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None and ("weight_hh" in name or "weight_ih" in name):
            total_sq += param.grad.data.norm(2).item() ** 2
    return total_sq ** 0.5


# ──────────────────────────────────────────────────────────────────────
#  Metrics Logger
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Container for one epoch's worth of logged values."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    grad_norm: float          # total recurrent gradient norm
    per_layer_norms: Dict[str, float] = field(default_factory=dict)


class MetricsLogger:
    """
    Accumulates EpochMetrics across training and writes them to CSV.

    Usage
    -----
    >>> logger = MetricsLogger()
    >>> logger.log(EpochMetrics(epoch=0, train_loss=1.2, ...))
    >>> logger.save_csv("results/rnn_seq100.csv")
    """

    def __init__(self) -> None:
        self.history: List[EpochMetrics] = []

    def log(self, metrics: EpochMetrics) -> None:
        self.history.append(metrics)

    @property
    def train_losses(self) -> List[float]:
        return [m.train_loss for m in self.history]

    @property
    def val_accs(self) -> List[float]:
        return [m.val_acc for m in self.history]

    @property
    def grad_norms(self) -> List[float]:
        return [m.grad_norm for m in self.history]

    def save_csv(self, path: str) -> None:
        """Persist full training history to a CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc", "grad_norm",
            ])
            for m in self.history:
                writer.writerow([
                    m.epoch, f"{m.train_loss:.6f}", f"{m.train_acc:.4f}",
                    f"{m.val_loss:.6f}", f"{m.val_acc:.4f}", f"{m.grad_norm:.6f}",
                ])

    def get_best_val_acc(self) -> float:
        """Return the best validation accuracy achieved."""
        if not self.history:
            return 0.0
        return max(m.val_acc for m in self.history)
