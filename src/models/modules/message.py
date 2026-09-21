"""Message Generation & Aggregation Modules for TGN."""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class MessageFunction(nn.Module):
    """Computes interaction message m_ij(t) = MLP([s_i, s_j, dt_i, dt_j, e_ij])."""

    def __init__(
        self,
        memory_dim: int = 64,
        edge_feat_dim: int = 8,
        time_dim: int = 1,
        message_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        # Input: s_src (mem_dim) + s_dst (mem_dim) + dt_src (1) + dt_dst (1) + edge_feats (edge_dim)
        input_dim = (2 * memory_dim) + (2 * time_dim) + edge_feat_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(message_dim, message_dim),
            nn.LayerNorm(message_dim),
        )

    def forward(
        self,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        delta_t_src: torch.Tensor,
        delta_t_dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes message for both source and destination nodes from an interaction."""
        # Ensure delta_t has 2D shape (B, 1)
        if delta_t_src.dim() == 1:
            delta_t_src = delta_t_src.unsqueeze(1)
        if delta_t_dst.dim() == 1:
            delta_t_dst = delta_t_dst.unsqueeze(1)
            
        # Concatenate features
        raw_input = torch.cat([src_memory, dst_memory, delta_t_src, delta_t_dst, edge_features], dim=1)
        
        # Message for source and receiver
        src_msg = self.mlp(raw_input)
        
        # Symmetrical message for destination with swapped memories
        raw_input_dst = torch.cat([dst_memory, src_memory, delta_t_dst, delta_t_src, edge_features], dim=1)
        dst_msg = self.mlp(raw_input_dst)
        
        return src_msg, dst_msg


class MessageAggregator(nn.Module):
    """Aggregates multiple messages arriving at the same node within a batch using Mean."""

    def __init__(self):
        super().__init__()

    def aggregate(
        self,
        node_ids: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Groups messages by unique node_id and computes mean aggregation.
        
        Returns:
            unique_nodes: (U,) unique node IDs
            aggregated_msgs: (U, msg_dim)
            latest_timestamps: (U,) latest timestamp for each node
        """
        unique_nodes, inverse_indices = torch.unique(node_ids, return_inverse=True)
        num_unique = len(unique_nodes)
        
        # Accumulate messages
        msg_dim = messages.size(1)
        agg_msgs = torch.zeros(num_unique, msg_dim, dtype=messages.dtype, device=messages.device)
        counts = torch.zeros(num_unique, 1, dtype=messages.dtype, device=messages.device)
        latest_ts = torch.zeros(num_unique, dtype=timestamps.dtype, device=timestamps.device)
        
        agg_msgs.index_add_(0, inverse_indices, messages)
        counts.index_add_(0, inverse_indices, torch.ones_like(messages[:, :1]))
        
        agg_msgs = agg_msgs / torch.clamp(counts, min=1.0)
        
        # Compute latest timestamp per unique node
        for i in range(len(node_ids)):
            u_idx = inverse_indices[i]
            if timestamps[i] > latest_ts[u_idx]:
                latest_ts[u_idx] = timestamps[i]
                
        return unique_nodes, agg_msgs, latest_ts
