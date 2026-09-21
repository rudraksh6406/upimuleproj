"""Temporal Graph Attention & Continuous Time Encoding for TGN."""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """Continuous Time Harmonic Encoding via Learnable Cosine/Sine Frequencies (Bochner's Theorem)."""

    def __init__(self, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        # Learnable frequency parameters initialized exponentially
        init_freqs = 1.0 / (10 ** np.linspace(0, 9, time_dim))
        self.freqs = nn.Parameter(torch.from_numpy(init_freqs).float())

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Encodes delta_t (B,) or (B, K) into (B, time_dim) or (B, K, time_dim)."""
        freqs = self.freqs.to(delta_t.device)
        if delta_t.dim() == 1:
            # (B, 1) * (1, time_dim) -> (B, time_dim)
            angles = delta_t.unsqueeze(-1) * freqs.unsqueeze(0)
        else:
            # (B, K, 1) * (1, 1, time_dim) -> (B, K, time_dim)
            angles = delta_t.unsqueeze(-1) * freqs.view(1, 1, -1)
        return torch.cos(angles)


class TemporalNeighborSampler:
    """Maintains a dynamic history of 1-hop temporal interactions for neighbor sampling."""

    def __init__(self, max_neighbors: int = 20):
        self.max_neighbors = max_neighbors
        # Map node_id -> list of tuples: (neighbor_id, timestamp, edge_feat_tensor)
        self.adj_list: Dict[int, List[Tuple[int, float, torch.Tensor]]] = {}

    def clear(self) -> None:
        self.adj_list.clear()

    def add_interaction(self, u: int, v: int, ts: float, edge_feat: torch.Tensor) -> None:
        """Records an undirected temporal interaction between u and v."""
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
            
        self.adj_list[u].append((v, ts, edge_feat.detach().cpu()))
        self.adj_list[v].append((u, ts, edge_feat.detach().cpu()))

    def get_temporal_neighbors(
        self,
        node_ids: List[int],
        current_ts: List[float],
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples up to K most recent temporal neighbors that interacted BEFORE current_ts.
        
        Returns:
            neighbor_ids: (B, K) int64
            delta_times: (B, K) float32
            edge_feats: (B, K, edge_feat_dim) float32
            mask: (B, K) bool (True for valid neighbors, False for padding)
        """
        k = k or self.max_neighbors
        b_size = len(node_ids)
        
        # We find feature dim from sample if available
        sample_feat_dim = 8
        for neighbors in self.adj_list.values():
            if neighbors:
                sample_feat_dim = neighbors[0][2].shape[-1]
                break

        nbr_ids = torch.zeros(b_size, k, dtype=torch.long)
        delta_ts = torch.zeros(b_size, k, dtype=torch.float32)
        edge_feats = torch.zeros(b_size, k, sample_feat_dim, dtype=torch.float32)
        mask = torch.zeros(b_size, k, dtype=torch.bool)

        for i, (n_id, curr_t) in enumerate(zip(node_ids, current_ts)):
            history = self.adj_list.get(n_id, [])
            # Filter strictly historical interactions
            valid = [item for item in history if item[1] <= curr_t]
            # Take most recent K
            recent = valid[-k:] if valid else []
            
            for j, (v_nbr, t_nbr, e_feat) in enumerate(recent):
                nbr_ids[i, j] = v_nbr
                delta_ts[i, j] = max(0.0, curr_t - t_nbr)
                edge_feats[i, j] = e_feat
                mask[i, j] = True

        return nbr_ids, delta_ts, edge_feats, mask


class TemporalGraphAttention(nn.Module):
    """Multi-Head Temporal Graph Attention Layer.
    Computes embedding z_i(t) and captures attention weights for explainability.
    """

    def __init__(
        self,
        memory_dim: int = 64,
        edge_dim: int = 8,
        time_dim: int = 32,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.time_encoder = TimeEncoder(time_dim)

        # Query projection (source node memory)
        self.w_q = nn.Linear(memory_dim, embedding_dim, bias=False)
        # Key & Value projection (neighbor memory + time enc + edge feat)
        kv_input_dim = memory_dim + time_dim + edge_dim
        self.w_k = nn.Linear(kv_input_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(kv_input_dim, embedding_dim, bias=False)

        self.out_proj = nn.Linear(embedding_dim + memory_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        src_memory: torch.Tensor,
        neighbor_memory: torch.Tensor,
        delta_ts: torch.Tensor,
        edge_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes attention aggregated embedding.
        
        Args:
            src_memory: (B, mem_dim)
            neighbor_memory: (B, K, mem_dim)
            delta_ts: (B, K)
            edge_feats: (B, K, edge_dim)
            mask: (B, K) boolean mask (True for actual neighbor)
            
        Returns:
            z: (B, embedding_dim) node embeddings
            attn_weights: (B, num_heads, K) attention weights for explainability
        """
        b_size, k, _ = neighbor_memory.shape
        
        # Time encoding for delta_t
        time_enc = self.time_encoder(delta_ts)  # (B, K, time_dim)
        
        # Query: (B, 1, num_heads, head_dim)
        q = self.w_q(src_memory).view(b_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Key and Value inputs: neighbor_mem + time_enc + edge_feats
        kv_input = torch.cat([neighbor_memory, time_enc, edge_feats], dim=-1)
        
        k_mat = self.w_k(kv_input).view(b_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        v_mat = self.w_v(kv_input).view(b_size, k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention scores
        # (B, H, 1, head_dim) @ (B, H, head_dim, K) -> (B, H, 1, K)
        scores = torch.matmul(q, k_mat.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Mask out non-existent padding neighbors
        # mask is (B, K) -> (B, 1, 1, K)
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~mask_expanded, -1e9)
        
        # Softmax over neighbors
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 1, K)
        attn_weights_clean = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply attention to values
        # (B, H, 1, K) @ (B, H, K, head_dim) -> (B, H, 1, head_dim)
        context = torch.matmul(self.dropout(attn_weights_clean), v_mat)
        context = context.transpose(1, 2).contiguous().view(b_size, self.embedding_dim)
        
        # Residual connection with source memory
        combined = torch.cat([context, src_memory], dim=-1)
        out = self.out_proj(combined)
        z = self.layer_norm(self.dropout(F.relu(out)))
        
        # Average attention across heads for explainability output: (B, K)
        avg_attn = attn_weights_clean.squeeze(2).mean(dim=1)
        
        return z, avg_attn
