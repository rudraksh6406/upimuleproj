"""Node Memory Module for Temporal Graph Neural Network (TGN)."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class NodeMemory(nn.Module):
    """Maintains and updates dynamic continuous-time memory states for all nodes."""

    def __init__(self, num_nodes: int, memory_dim: int = 64, message_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.device = device

        # GRU Cell for memory updates
        self.gru_cell = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)
        
        # State buffers (not model parameters, but persistent state)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim, device=device))
        self.register_buffer("last_update_ts", torch.zeros(num_nodes, device=device))
        
        # Raw message store buffer for nodes involved in events
        self.messages: Dict[int, list] = {}

    def reset_state(self) -> None:
        """Resets memory state and timestamps to zero (e.g., at the start of epoch)."""
        self.memory.zero_()
        self.last_update_ts.zero_()
        self.messages.clear()

    def detach_memory(self) -> None:
        """Detaches memory tensors from computational graph to allow BPTT across batches."""
        self.memory = self.memory.detach()

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Returns memory vector s_i(t) for given node indices."""
        return self.memory[node_ids]

    def get_last_update(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Returns last update timestamps for given node indices."""
        return self.last_update_ts[node_ids]

    def store_raw_messages(
        self,
        unique_nodes: torch.Tensor,
        aggregated_messages: torch.Tensor,
        timestamps: torch.Tensor
    ) -> None:
        """Stores aggregated messages to be applied in memory update."""
        for idx, n_id in enumerate(unique_nodes.tolist()):
            self.messages[n_id] = (aggregated_messages[idx], timestamps[idx])

    def update_nodes_memory(self, node_ids: torch.Tensor, aggregated_messages: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Applies GRU update: s_i(t) = GRU(M_i(t), s_i(t^-))."""
        prev_mem = self.memory[node_ids]
        updated_mem = self.gru_cell(aggregated_messages, prev_mem)
        
        # In-place write back to state
        self.memory[node_ids] = updated_mem
        self.last_update_ts[node_ids] = timestamps
        return updated_mem
