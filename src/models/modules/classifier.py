"""Mule Classification Head for TGN."""
import torch
import torch.nn as nn
from typing import List


class MuleClassifier(nn.Module):
    """3-Layer MLP Classifier producing mule account probabilities."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        dims = hidden_dims or [64, 32, 1]
        
        layers = []
        prev_dim = input_dim
        
        for idx, h_dim in enumerate(dims[:-1]):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Final projection to 1 logit
        layers.append(nn.Linear(prev_dim, dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Outputs probabilities P(mule | u, t) via Sigmoid."""
        logits = self.mlp(embeddings)
        return torch.sigmoid(logits).squeeze(-1)

    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Outputs raw logits for BCEWithLogitsLoss."""
        return self.mlp(embeddings).squeeze(-1)
