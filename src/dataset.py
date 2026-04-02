"""
dataset.py — Synthetic Long-Sequence Memory Stress Dataset

Generates sequences where the model must remember information from the
FIRST timestep across an arbitrarily long noisy sequence to predict the
correct output.  This directly tests long-term dependency retention.

Two task variants are supported:
  1. "copy_first"  — Classify the value of the first element (binary).
  2. "delayed_sum" — Predict a continuous target that depends on the
                     first K elements, buried under noise.

Author : GRU-vs-LSTM Research Project
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple, Dict, List


# ──────────────────────────────────────────────────────────────────────
#  Core Dataset
# ──────────────────────────────────────────────────────────────────────

class MemoryStressDataset(Dataset):
    """
    Synthetic dataset designed to stress-test recurrent memory.

    Each sample is a sequence of length `seq_len`.  The first element is
    a *signal* drawn from {0, 1, …, num_classes-1}.  All remaining
    elements are Gaussian noise (mean=0, std=noise_std).  The label for
    the entire sequence equals the signal element.

    The model therefore **must** retain the first-timestep information
    across `seq_len - 1` noisy steps to classify correctly — a direct
    test of long-term dependency learning.

    Parameters
    ----------
    num_samples : int
        Total number of sequences to generate.
    seq_len : int
        Length of each sequence (number of timesteps).
    num_classes : int
        Number of distinct classes for the signal element (default 8).
    noise_std : float
        Standard deviation of the Gaussian distractor noise.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10_000,
        seq_len: int = 100,
        num_classes: int = 8,
        noise_std: float = 1.0,
        seed: int | None = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.noise_std = noise_std

        rng = np.random.default_rng(seed)

        # --- build sequences ---------------------------------------------------
        # Shape: (num_samples, seq_len, 1)  — single-feature time series
        self.sequences = rng.normal(
            loc=0.0, scale=noise_std, size=(num_samples, seq_len, 1)
        ).astype(np.float32)

        # The signal: first timestep carries the class information
        self.labels = rng.integers(0, num_classes, size=num_samples)
        self.sequences[:, 0, 0] = self.labels.astype(np.float32)

        # Convert to tensors
        self.sequences = torch.from_numpy(self.sequences)
        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ──────────────────────────────────────────────────────────────────────
#  Convenience Builders
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(
    seq_len: int,
    batch_size: int = 64,
    num_samples: int = 10_000,
    num_classes: int = 8,
    noise_std: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for a given sequence length.

    Returns
    -------
    train_loader, test_loader
    """
    dataset = MemoryStressDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        num_classes=num_classes,
        noise_std=noise_std,
        seed=seed,
    )
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


def build_all_dataloaders(
    sequence_lengths: List[int],
    batch_size: int = 64,
    num_samples: int = 10_000,
    num_classes: int = 8,
    noise_std: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """
    Build dataloaders for every requested sequence length.

    Returns
    -------
    dict  —  {seq_len: (train_loader, test_loader)}
    """
    loaders: Dict[int, Tuple[DataLoader, DataLoader]] = {}
    for sl in sequence_lengths:
        loaders[sl] = build_dataloaders(
            seq_len=sl,
            batch_size=batch_size,
            num_samples=num_samples,
            num_classes=num_classes,
            noise_std=noise_std,
            train_ratio=train_ratio,
            seed=seed,
        )
    return loaders


# ──────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = MemoryStressDataset(num_samples=5, seq_len=20, num_classes=4)
    for i in range(len(ds)):
        seq, lbl = ds[i]
        print(f"Sample {i}: first_elem={seq[0, 0].item():.1f}  label={lbl.item()}  seq_shape={seq.shape}")
