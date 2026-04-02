"""
models.py — Recurrent Architectures for the Memory Stress Benchmark

Implements four models with a **unified interface** so that the training
pipeline can treat them interchangeably:

  1. VanillaRNN   — standard Elman RNN (known to suffer vanishing gradients)
  2. GRUModel     — Gated Recurrent Unit
  3. LSTMModel    — Long Short-Term Memory
  4. AttnLSTM     — LSTM + temporal self-attention  [NOVELTY]

Design decisions
----------------
* All models share the same hidden_size, num_layers, and dropout for a
  **fair, controlled comparison**.
* Orthogonal weight initialisation is applied to recurrent kernels —
  a well-known technique for stabilising gradient flow in deep RNNs
  (Saxe et al., 2014).  This lets us study whether the *architecture*
  (not just initialisation) drives gradient behaviour.
* The forward() method returns `(logits, last_hidden)` so the trainer
  can optionally inspect hidden states.

Author : GRU-vs-LSTM Research Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
#  Shared weight initialisation
# ──────────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module) -> None:
    """Apply orthogonal init to recurrent weights, Xavier to linear."""
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)


# ══════════════════════════════════════════════════════════════════════
#  1. Vanilla RNN
# ══════════════════════════════════════════════════════════════════════

class VanillaRNN(nn.Module):
    """Standard Elman RNN — expected to fail on long sequences."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = "VanillaRNN"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)          # out: (B, T, H)
        logits = self.fc(out[:, -1, :]) # use last timestep
        return logits, h_n


# ══════════════════════════════════════════════════════════════════════
#  2. GRU
# ══════════════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """Gated Recurrent Unit — lighter gating than LSTM."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = "GRU"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h_n = self.gru(x)
        logits = self.fc(out[:, -1, :])
        return logits, h_n


# ══════════════════════════════════════════════════════════════════════
#  3. LSTM
# ══════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """Long Short-Term Memory — gold standard for long-range deps."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, (h_n, c_n) = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits, h_n


# ══════════════════════════════════════════════════════════════════════
#  4. Attention-Enhanced LSTM  [NOVELTY]
# ══════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """
    Bahdanau-style additive attention over LSTM hidden states.

    Instead of relying solely on the last hidden state, attention lets
    the classifier *look back* at every timestep and focus on the most
    informative one — theoretically the first timestep in our task.

    This provides a research baseline to show how attention mitigates
    the memory bottleneck that plagues vanilla architectures.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        rnn_outputs : (batch, seq_len, hidden_size)

        Returns
        -------
        context : (batch, hidden_size)
        weights : (batch, seq_len)   — attention weights (for viz)
        """
        scores = self.attn(rnn_outputs).squeeze(-1)      # (B, T)
        weights = F.softmax(scores, dim=-1)               # (B, T)
        context = torch.bmm(weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context, weights


class AttnLSTM(nn.Module):
    """
    LSTM + Temporal Attention — a modern approach that should excel
    even on very long sequences, since attention can directly access
    early timesteps without gradient propagation through all T steps.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_name = "Attn-LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, (h_n, c_n) = self.lstm(x)              # out: (B, T, H)
        context, attn_weights = self.attention(out)  # (B, H), (B, T)
        logits = self.fc(context)
        return logits, attn_weights   # return attn weights for analysis


# ══════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "RNN":       VanillaRNN,
    "GRU":       GRUModel,
    "LSTM":      LSTMModel,
    "Attn-LSTM": AttnLSTM,
}


def build_model(
    name: str,
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 8,
    dropout: float = 0.0,
) -> nn.Module:
    """Instantiate a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


# ──────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch, seq, feat = 4, 100, 1
    x = torch.randn(batch, seq, feat)

    for name in MODEL_REGISTRY:
        model = build_model(name)
        logits, extra = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:10s} | logits={logits.shape}  params={n_params:,}")
