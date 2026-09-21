"""Real-Time Inference Engine for MuleGuard."""
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from ..data.preprocessor import GraphPreprocessor
from ..models.tgn import TGNModel
from ..utils.timer import Timer


class RealTimeInferenceEngine:
    """Real-time scoring worker processing transaction streams using a trained TGN model."""

    def __init__(
        self,
        model: TGNModel,
        preprocessor: GraphPreprocessor,
        device: str = "cpu",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.timer = Timer()

    def process_transaction(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Scores an individual transaction event in real time (<50ms) and returns predictions.
        
        Returns:
            Dict containing sender score, receiver score, event score, top attention neighbors, latency_ms.
        """
        self.timer.start()
        
        u_vpa = txn["sender_vpa"]
        v_vpa = txn["receiver_vpa"]
        ts = float(txn.get("timestamp", time.time()))
        
        u_id = self.preprocessor.get_or_create_node_id(u_vpa)
        v_id = self.preprocessor.get_or_create_node_id(v_vpa)
        
        edge_feat = self.preprocessor.extract_edge_features(txn, u_id, v_id)
        
        # Convert to tensors
        src_t = torch.tensor([u_id], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([v_id], dtype=torch.long, device=self.device)
        ts_t = torch.tensor([ts], dtype=torch.float32, device=self.device)
        ef_t = torch.from_numpy(edge_feat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model.forward_events(src_t, dst_t, ts_t, ef_t)
            
            p_src = float(out["p_src"][0].item())
            p_dst = float(out["p_dst"][0].item())
            p_event = float(out["p_event"][0].item())
            
            # Extract top attention neighbors for receiver and sender
            nbr_ids_src, _, _, mask_src = self.model.neighbor_sampler.get_temporal_neighbors([u_id], [ts])
            top_src_nbrs = []
            if mask_src.any():
                attn_s = out["attn_src"][0]
                for j in range(nbr_ids_src.shape[1]):
                    if mask_src[0, j]:
                        n_j = int(nbr_ids_src[0, j].item())
                        vpa_j = self.preprocessor.idx_to_node.get(n_j, f"acc_{n_j}")
                        top_src_nbrs.append({
                            "vpa": vpa_j,
                            "attention_weight": round(float(attn_s[j].item()), 4),
                        })

            nbr_ids_dst, _, _, mask_dst = self.model.neighbor_sampler.get_temporal_neighbors([v_id], [ts])
            top_dst_nbrs = []
            if mask_dst.any():
                attn_d = out["attn_dst"][0]
                for j in range(nbr_ids_dst.shape[1]):
                    if mask_dst[0, j]:
                        n_j = int(nbr_ids_dst[0, j].item())
                        vpa_j = self.preprocessor.idx_to_node.get(n_j, f"acc_{n_j}")
                        top_dst_nbrs.append({
                            "vpa": vpa_j,
                            "attention_weight": round(float(attn_d[j].item()), 4),
                        })

        latency_ms = self.timer.stop()

        return {
            "txn_id": txn.get("txn_id", ""),
            "sender_vpa": u_vpa,
            "receiver_vpa": v_vpa,
            "sender_score": round(p_src, 4),
            "receiver_score": round(p_dst, 4),
            "event_score": round(p_event, 4),
            "top_sender_neighbors": top_src_nbrs[:5],
            "top_receiver_neighbors": top_dst_nbrs[:5],
            "latency_ms": round(latency_ms, 2),
            "timestamp": ts,
        }
