"""
trainer.py — Training Pipeline with Gradient Dynamics Tracking

Core responsibilities:
  1. Standard supervised training loop (forward → loss → backward → step)
  2. **Gradient hook injection** — after every backward pass we capture
     the L2-norm of each recurrent weight's gradient.  This is the
     mechanism that lets us *prove* vanishing / exploding gradients.
  3. Validation evaluation at the end of every epoch.
  4. Logging every metric to a MetricsLogger for downstream plotting.

The trainer is model-agnostic: any nn.Module with a .forward() that
returns (logits, extra) can be passed in.

Author : GRU-vs-LSTM Research Project
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional

from src.metrics import (
    compute_accuracy,
    compute_gradient_norms,
    compute_total_gradient_norm,
    EpochMetrics,
    MetricsLogger,
)


# ──────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_value: float = 5.0,
) -> Tuple[float, float, float, dict]:
    """
    Run one full training epoch.

    Returns
    -------
    avg_loss, avg_acc, total_grad_norm, per_layer_norms
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    epoch_grad_norm = 0.0
    epoch_per_layer: dict = {}

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        # ── Gradient analysis (this is the CORE research contribution) ──
        batch_norm = compute_total_gradient_norm(model)
        batch_per_layer = compute_gradient_norms(model)
        epoch_grad_norm += batch_norm

        # Accumulate per-layer norms for averaging
        for k, v in batch_per_layer.items():
            epoch_per_layer[k] = epoch_per_layer.get(k, 0.0) + v

        # Gradient clipping — prevents NaN explosions but still
        # lets us *observe* the gradient magnitude before clipping.
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item()
        total_acc += compute_accuracy(logits, labels)
        n_batches += 1

    # Average over batches
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_grad_norm = epoch_grad_norm / n_batches
    avg_per_layer = {k: v / n_batches for k, v in epoch_per_layer.items()}

    return avg_loss, avg_acc, avg_grad_norm, avg_per_layer


# ──────────────────────────────────────────────────────────────────────
#  Validation Loop
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run validation; returns (avg_loss, avg_accuracy)."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        logits, _ = model(sequences)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, labels)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


# ──────────────────────────────────────────────────────────────────────
#  Full Training Routine
# ──────────────────────────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    clip_value: float = 5.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> MetricsLogger:
    """
    Full training run for one model on one sequence length.

    Parameters
    ----------
    model : nn.Module
        Must return (logits, extra) from forward().
    train_loader, test_loader : DataLoader
    epochs : int
    lr : float
    clip_value : float
        Max gradient norm for clipping (applied AFTER measurement).
    device : torch.device | None
        Auto-detects CUDA if available.
    verbose : bool

    Returns
    -------
    MetricsLogger with full epoch-by-epoch history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger = MetricsLogger()
    model_name = getattr(model, "model_name", model.__class__.__name__)

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {model_name}", unit="epoch")

    for epoch in iterator:
        # --- Train ---
        train_loss, train_acc, grad_norm, per_layer = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_value,
        )

        # --- Validate ---
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        # --- Log ---
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            grad_norm=grad_norm,
            per_layer_norms=per_layer,
        )
        logger.log(metrics)

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{train_loss:.4f}",
                acc=f"{val_acc:.2%}",
                grad=f"{grad_norm:.4f}",
            )

    return logger
