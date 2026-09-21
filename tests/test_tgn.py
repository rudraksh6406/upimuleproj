"""Unit tests for TGN neural network modules and full model."""
import torch
import pytest
from src.models.tgn import TGNModel
from src.models.modules.memory import NodeMemory
from src.models.modules.attention import TimeEncoder, TemporalGraphAttention, TemporalNeighborSampler


def test_time_encoder():
    te = TimeEncoder(time_dim=32)
    dt = torch.tensor([10.0, 50.0, 3600.0])
    out = te(dt)
    assert out.shape == (3, 32)


def test_node_memory():
    mem = NodeMemory(num_nodes=10, memory_dim=64, message_dim=64)
    nodes = torch.tensor([1, 3])
    init_mem = mem.get_memory(nodes)
    assert init_mem.shape == (2, 64)
    assert torch.all(init_mem == 0.0)


def test_tgn_forward_pass():
    model = TGNModel(num_nodes=50, memory_dim=32, embedding_dim=64, message_dim=32)
    src = torch.tensor([0, 1, 2], dtype=torch.long)
    dst = torch.tensor([3, 4, 5], dtype=torch.long)
    ts = torch.tensor([100.0, 105.0, 110.0], dtype=torch.float32)
    edge_feats = torch.randn(3, 8)
    labels = torch.tensor([0.0, 1.0, 0.0])

    out = model.forward_events(src, dst, ts, edge_feats, labels=labels)
    assert "p_src" in out
    assert "p_dst" in out
    assert "p_event" in out
    assert out["loss"] is not None
    assert out["p_event"].shape == (3,)
