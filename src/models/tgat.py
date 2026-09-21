"""TGAT Baseline (Temporal Graph Attention without Memory Module)."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .tgn import TGNModel


class TGATBaseline(TGNModel):
    """TGAT Baseline: Same temporal graph attention and time encoding, but use_memory is disabled."""

    def __init__(self, num_nodes: int, **kwargs):
        kwargs["use_memory"] = False
        super().__init__(num_nodes=num_nodes, **kwargs)
