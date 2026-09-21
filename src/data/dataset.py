"""PyTorch Dataset & Temporal Batch Loader for TGN."""
import math
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessor import GraphPreprocessor


class TemporalGraphData:
    """Container for processed temporal graph events."""
    def __init__(
        self,
        src_nodes: np.ndarray,
        dst_nodes: np.ndarray,
        timestamps: np.ndarray,
        edge_feats: np.ndarray,
        labels: np.ndarray,
        node_to_idx: Dict[str, int],
        idx_to_node: Dict[int, str],
        mule_vpas: Set[str],
    ):
        self.src_nodes = src_nodes          # (E,) int64
        self.dst_nodes = dst_nodes          # (E,) int64
        self.timestamps = timestamps        # (E,) float32
        self.edge_feats = edge_feats        # (E, F) float32
        self.labels = labels                # (E,) int64 (1 if dst or src is mule)
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node
        self.mule_vpas = mule_vpas
        self.num_nodes = len(node_to_idx)
        self.num_events = len(src_nodes)


def build_temporal_graph_data(
    txns: List[Dict[str, Any]],
    mule_vpas: Set[str],
    preprocessor: Optional[GraphPreprocessor] = None,
) -> TemporalGraphData:
    """Converts a raw transaction list into indexed numpy arrays for TGN."""
    prep = preprocessor or GraphPreprocessor()
    
    src_list, dst_list, ts_list, feat_list, label_list = [], [], [], [], []

    for txn in txns:
        u_vpa = txn["sender_vpa"]
        v_vpa = txn["receiver_vpa"]
        ts = float(txn["timestamp"])
        
        u_id = prep.get_or_create_node_id(u_vpa)
        v_id = prep.get_or_create_node_id(v_vpa)
        
        edge_feat = prep.extract_edge_features(txn, u_id, v_id)
        
        # Label is 1 if either sender or receiver is mule
        is_mule = 1 if (u_vpa in mule_vpas or v_vpa in mule_vpas or txn.get("is_mule_sender") or txn.get("is_mule_receiver")) else 0
        
        src_list.append(u_id)
        dst_list.append(v_id)
        ts_list.append(ts)
        feat_list.append(edge_feat)
        label_list.append(is_mule)

    return TemporalGraphData(
        src_nodes=np.array(src_list, dtype=np.int64),
        dst_nodes=np.array(dst_list, dtype=np.int64),
        timestamps=np.array(ts_list, dtype=np.float32),
        edge_feats=np.array(feat_list, dtype=np.float32),
        labels=np.array(label_list, dtype=np.int64),
        node_to_idx=prep.node_to_idx,
        idx_to_node=prep.idx_to_node,
        mule_vpas=mule_vpas,
    )


class TemporalBatchLoader:
    """Iterates through temporal graph events in chronological batches."""

    def __init__(self, data: TemporalGraphData, batch_size: int = 200):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = math.ceil(data.num_events / batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for i in range(0, self.data.num_events, self.batch_size):
            end = min(i + self.batch_size, self.data.num_events)
            
            src = torch.from_numpy(self.data.src_nodes[i:end])
            dst = torch.from_numpy(self.data.dst_nodes[i:end])
            ts = torch.from_numpy(self.data.timestamps[i:end])
            edge_feats = torch.from_numpy(self.data.edge_feats[i:end])
            labels = torch.from_numpy(self.data.labels[i:end]).float()
            
            yield {
                "src": src,
                "dst": dst,
                "timestamps": ts,
                "edge_feats": edge_feats,
                "labels": labels,
                "batch_idx": i // self.batch_size,
            }
