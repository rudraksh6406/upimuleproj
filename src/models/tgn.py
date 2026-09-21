"""Temporal Graph Neural Network (TGN) for Mule Account Detection."""
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.attention import TemporalGraphAttention, TemporalNeighborSampler
from .modules.classifier import MuleClassifier
from .modules.memory import NodeMemory
from .modules.message import MessageAggregator, MessageFunction


class TGNModel(nn.Module):
    """5-Module Temporal Graph Neural Network Architecture for UPI Mule Detection."""

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 64,
        embedding_dim: int = 128,
        message_dim: int = 64,
        edge_feat_dim: int = 8,
        time_dim: int = 32,
        attention_heads: int = 4,
        dropout: float = 0.2,
        neighborhood_size: int = 20,
        device: str = "cpu",
        use_memory: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_memory = use_memory
        self.use_attention = use_attention
        self.neighborhood_size = neighborhood_size

        # 1. Node Memory Module
        self.memory = NodeMemory(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=message_dim,
            device=device,
        )

        # 2. Message Function MLP
        self.message_func = MessageFunction(
            memory_dim=memory_dim,
            edge_feat_dim=edge_feat_dim,
            time_dim=1,
            message_dim=message_dim,
            dropout=dropout,
        )

        # 3. Message Aggregator
        self.msg_aggregator = MessageAggregator()

        # 4. Temporal Neighbor Sampler (dynamic historical graph)
        self.neighbor_sampler = TemporalNeighborSampler(max_neighbors=neighborhood_size)

        # 5. Temporal Graph Attention (Embedding)
        self.attention = TemporalGraphAttention(
            memory_dim=memory_dim,
            edge_dim=edge_feat_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            num_heads=attention_heads,
            dropout=dropout,
        )

        # 6. Classification Head
        self.classifier = MuleClassifier(
            input_dim=embedding_dim,
            hidden_dims=[64, 32, 1],
            dropout=dropout,
        )

        self.bce_loss = nn.BCELoss()

    def reset_state(self) -> None:
        """Resets memory state and neighbor interactions."""
        self.memory.reset_state()
        self.neighbor_sampler.clear()

    def detach_memory(self) -> None:
        """Detaches memory graph for truncated BPTT."""
        self.memory.detach_memory()

    def compute_temporal_embeddings(
        self,
        node_ids: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes continuous dynamic embeddings z_i(t) for a batch of nodes."""
        node_list = node_ids.tolist()
        ts_list = timestamps.tolist()

        # Step 1: Retrieve source node memory s_i(t)
        src_mem = self.memory.get_memory(node_ids)  # (B, mem_dim)

        if not self.use_attention:
            # Fallback simple linear embedding if attention disabled (ablation)
            z = src_mem
            return z, torch.zeros(len(node_ids), self.neighborhood_size, device=src_mem.device)

        # Step 2: Sample temporal historical neighbors
        nbr_ids, delta_ts, edge_feats, mask = self.neighbor_sampler.get_temporal_neighbors(
            node_list, ts_list, k=self.neighborhood_size
        )
        nbr_ids = nbr_ids.to(self.device)
        delta_ts = delta_ts.to(self.device)
        edge_feats = edge_feats.to(self.device)
        mask = mask.to(self.device)

        # Step 3: Fetch neighbor memories
        # (B, K, mem_dim)
        nbr_memory = self.memory.get_memory(nbr_ids)

        # Step 4: Temporal Multi-Head Attention
        z, attn_weights = self.attention(
            src_memory=src_mem,
            neighbor_memory=nbr_memory,
            delta_ts=delta_ts,
            edge_feats=edge_feats,
            mask=mask,
        )

        return z, attn_weights

    def forward_events(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamps: torch.Tensor,
        edge_feats: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Processes a batch of temporal interaction events, generates predictions, and updates state."""
        src = src.to(self.device)
        dst = dst.to(self.device)
        timestamps = timestamps.to(self.device)
        edge_feats = edge_feats.to(self.device)

        # 1. Compute dynamic node embeddings before the interaction takes effect
        z_src, attn_src = self.compute_temporal_embeddings(src, timestamps)
        z_dst, attn_dst = self.compute_temporal_embeddings(dst, timestamps)

        # 2. Predict mule probabilities
        p_src = self.classifier(z_src)
        p_dst = self.classifier(z_dst)

        # Joint prediction for the interaction (max of src and dst)
        p_event = torch.maximum(p_src, p_dst)

        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.bce_loss(p_event, labels)

        # 3. Compute interaction messages and update memory if memory enabled
        if self.use_memory:
            last_ts_src = self.memory.get_last_update(src)
            last_ts_dst = self.memory.get_last_update(dst)
            dt_src = torch.clamp(timestamps - last_ts_src, min=0.0)
            dt_dst = torch.clamp(timestamps - last_ts_dst, min=0.0)

            src_mem = self.memory.get_memory(src)
            dst_mem = self.memory.get_memory(dst)

            src_msg, dst_msg = self.message_func(src_mem, dst_mem, dt_src, dt_dst, edge_feats)

            # Combine all node IDs and messages in this batch
            all_nodes = torch.cat([src, dst], dim=0)
            all_msgs = torch.cat([src_msg, dst_msg], dim=0)
            all_ts = torch.cat([timestamps, timestamps], dim=0)

            # Aggregate and update GRU memory
            u_nodes, agg_msgs, latest_ts = self.msg_aggregator.aggregate(all_nodes, all_msgs, all_ts)
            self.memory.update_nodes_memory(u_nodes, agg_msgs, latest_ts)

        # 4. Store interactions into temporal neighbor sampler
        for i in range(len(src)):
            u_i = int(src[i].item())
            v_i = int(dst[i].item())
            t_i = float(timestamps[i].item())
            e_i = edge_feats[i]
            self.neighbor_sampler.add_interaction(u_i, v_i, t_i, e_i)

        return {
            "p_src": p_src,
            "p_dst": p_dst,
            "p_event": p_event,
            "loss": loss,
            "attn_src": attn_src,
            "attn_dst": attn_dst,
            "z_src": z_src,
            "z_dst": z_dst,
        }

    def predict_account_score(
        self,
        node_id: int,
        timestamp: float,
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Inference helper to score a single account and retrieve its top attention neighbors."""
        n_tensor = torch.tensor([node_id], dtype=torch.long, device=self.device)
        t_tensor = torch.tensor([timestamp], dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            z, attn_weights = self.compute_temporal_embeddings(n_tensor, t_tensor)
            prob = float(self.classifier(z).item())

            # Retrieve neighbor IDs
            nbr_ids, _, _, mask = self.neighbor_sampler.get_temporal_neighbors(
                [node_id], [timestamp], k=self.neighborhood_size
            )
            top_neighbors = []
            for j in range(nbr_ids.shape[1]):
                if mask[0, j]:
                    nbr_id = int(nbr_ids[0, j].item())
                    weight = float(attn_weights[0, j].item())
                    top_neighbors.append((nbr_id, weight))

            # Sort by attention weight descending
            top_neighbors.sort(key=lambda x: x[1], reverse=True)

        return prob, top_neighbors[:5]
