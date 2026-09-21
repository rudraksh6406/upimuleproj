"""Static Graph Neural Network Baselines (GCN and GraphSAGE)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGCN(nn.Module):
    """2-Layer Graph Convolutional Network operating on static graph snapshots."""

    def __init__(self, in_features: int = 18, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, 32)
        self.classifier = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalized adjacency matrix: A_hat @ X @ W."""
        # Layer 1
        h1 = torch.matmul(adj_matrix, x)
        h1 = F.relu(self.conv1(h1))
        h1 = self.dropout(h1)

        # Layer 2
        h2 = torch.matmul(adj_matrix, h1)
        h2 = F.relu(self.conv2(h2))
        h2 = self.dropout(h2)

        # Output probabilities
        logits = self.classifier(h2).squeeze(-1)
        return torch.sigmoid(logits)


class StaticGraphSAGE(nn.Module):
    """GraphSAGE baseline with Mean Neighborhood Aggregator."""

    def __init__(self, in_features: int = 18, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.w_self1 = nn.Linear(in_features, hidden_dim)
        self.w_neigh1 = nn.Linear(in_features, hidden_dim)
        
        self.w_self2 = nn.Linear(hidden_dim, 32)
        self.w_neigh2 = nn.Linear(hidden_dim, 32)
        
        self.classifier = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Neighborhood mean aggregation + self transformation."""
        # Deg normalization for mean aggregation
        deg = torch.clamp(adj_matrix.sum(dim=-1, keepdim=True), min=1.0)
        norm_adj = adj_matrix / deg

        # Layer 1
        h_neigh1 = torch.matmul(norm_adj, x)
        h1 = F.relu(self.w_self1(x) + self.w_neigh1(h_neigh1))
        h1 = self.dropout(h1)

        # Layer 2
        h_neigh2 = torch.matmul(norm_adj, h1)
        h2 = F.relu(self.w_self2(h1) + self.w_neigh2(h_neigh2))
        h2 = self.dropout(h2)

        logits = self.classifier(h2).squeeze(-1)
        return torch.sigmoid(logits)
