"""Feature Engineering & Graph Preprocessing for MuleGuard."""
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import networkx as nx


class GraphPreprocessor:
    """Preprocesses raw UPI transactions into node and edge feature representations."""

    def __init__(self):
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        self.next_node_id: int = 0
        
        # State tracking for dynamic features
        self.node_last_ts: Dict[int, float] = {}
        self.node_txn_counts: Dict[int, int] = {}
        self.node_total_amounts: Dict[int, float] = {}
        self.known_node_devices: Dict[int, Set[str]] = {}

    def get_or_create_node_id(self, vpa: str) -> int:
        """Assigns a stable integer ID to a VPA."""
        if vpa not in self.node_to_idx:
            n_id = self.next_node_id
            self.node_to_idx[vpa] = n_id
            self.idx_to_node[n_id] = vpa
            self.next_node_id += 1
            self.node_last_ts[n_id] = 0.0
            self.node_txn_counts[n_id] = 0
            self.node_total_amounts[n_id] = 0.0
            self.known_node_devices[n_id] = set()
            return n_id
        return self.node_to_idx[vpa]

    def extract_edge_features(self, txn: Dict[str, Any], u_id: int, v_id: int) -> np.ndarray:
        """Extracts numerical edge features vector [dim = 8]:
        1. log1p(amount)
        2. delta_t_src (seconds since sender's last txn)
        3. delta_t_dst (seconds since receiver's last txn)
        4. is_p2m (0 for P2P, 1 for P2M)
        5. is_new_device (1 if sender device unseen, else 0)
        6. hour_of_day / 24.0 (cyclical time signal)
        7. day_of_week / 7.0
        8. sender_min_kyc (1 if min kyc, 0 if full)
        """
        amt = float(txn.get("amount", 0.0))
        ts = float(txn.get("timestamp", 0.0))
        
        last_u_ts = self.node_last_ts.get(u_id, ts)
        last_v_ts = self.node_last_ts.get(v_id, ts)
        delta_u = max(0.0, ts - last_u_ts)
        delta_v = max(0.0, ts - last_v_ts)
        
        dev_id = txn.get("device_id", "")
        is_new_dev = 1.0 if (dev_id and dev_id not in self.known_node_devices[u_id]) else 0.0
        if dev_id:
            self.known_node_devices[u_id].add(dev_id)
            
        # Update state
        self.node_last_ts[u_id] = ts
        self.node_last_ts[v_id] = ts
        self.node_txn_counts[u_id] = self.node_txn_counts.get(u_id, 0) + 1
        self.node_txn_counts[v_id] = self.node_txn_counts.get(v_id, 0) + 1
        self.node_total_amounts[u_id] = self.node_total_amounts.get(u_id, 0.0) + amt
        self.node_total_amounts[v_id] = self.node_total_amounts.get(v_id, 0.0) + amt

        # Time encoding
        hour = (ts % 86400.0) / 3600.0
        day = ((ts // 86400) % 7)

        feat = np.array([
            math.log1p(amt),
            math.log1p(delta_u),
            math.log1p(delta_v),
            1.0 if txn.get("txn_type") == "P2M" else 0.0,
            is_new_dev,
            hour / 24.0,
            day / 7.0,
            1.0 if txn.get("sender_kyc_type") == "min" else 0.0,
        ], dtype=np.float32)

        return feat

    def extract_tabular_features(
        self,
        txns: List[Dict[str, Any]],
        accounts: List[Dict[str, Any]],
        mule_vpas: Set[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extracts engineered node-level tabular and graph features for baseline models (XGBoost, LogReg, Rule-based).
        
        Features per node:
        - in_degree, out_degree, total_degree
        - total_in_amount, total_out_amount, net_flow
        - avg_in_amount, avg_out_amount, amount_ratio
        - unique_in_counterparts, unique_out_counterparts
        - burst_velocity_max (max txns in 1 hour)
        - dormancy_days
        - device_count
        - PageRank, Betweenness centrality (on graph snapshot)
        - is_merchant, is_min_kyc
        """
        G = nx.DiGraph()
        node_stats: Dict[str, Dict[str, Any]] = {}
        
        for acc in accounts:
            vpa = acc["vpa"]
            node_stats[vpa] = {
                "in_txns": 0,
                "out_txns": 0,
                "in_amount": 0.0,
                "out_amount": 0.0,
                "in_cps": set(),
                "out_cps": set(),
                "devices": set(acc.get("devices", [])),
                "timestamps": [],
                "amounts": [],
                "is_merchant": 1.0 if acc.get("is_merchant") else 0.0,
                "is_min_kyc": 1.0 if acc.get("kyc_type") == "min" else 0.0,
                "dormancy_days": float(acc.get("dormancy_days", 0.0)),
                "age_days": float(acc.get("account_age_days", 100.0)),
            }
            G.add_node(vpa)

        for txn in txns:
            u, v = txn["sender_vpa"], txn["receiver_vpa"]
            amt, t = float(txn["amount"]), float(txn["timestamp"])
            
            if u not in node_stats:
                node_stats[u] = {"in_txns": 0, "out_txns": 0, "in_amount": 0.0, "out_amount": 0.0,
                                 "in_cps": set(), "out_cps": set(), "devices": set(), "timestamps": [], "amounts": [],
                                 "is_merchant": 0.0, "is_min_kyc": 0.0, "dormancy_days": 0.0, "age_days": 100.0}
            if v not in node_stats:
                node_stats[v] = {"in_txns": 0, "out_txns": 0, "in_amount": 0.0, "out_amount": 0.0,
                                 "in_cps": set(), "out_cps": set(), "devices": set(), "timestamps": [], "amounts": [],
                                 "is_merchant": 0.0, "is_min_kyc": 0.0, "dormancy_days": 0.0, "age_days": 100.0}
                
            node_stats[u]["out_txns"] += 1
            node_stats[u]["out_amount"] += amt
            node_stats[u]["out_cps"].add(v)
            node_stats[u]["timestamps"].append(t)
            node_stats[u]["amounts"].append(amt)

            node_stats[v]["in_txns"] += 1
            node_stats[v]["in_amount"] += amt
            node_stats[v]["in_cps"].add(u)
            node_stats[v]["timestamps"].append(t)
            node_stats[v]["amounts"].append(amt)

            G.add_edge(u, v, weight=amt)

        # Compute graph metrics (with fallback for performance on large graphs)
        try:
            pagerank = nx.pagerank(G, max_iter=50)
        except Exception:
            pagerank = {n: 0.0 for n in G.nodes()}

        vpas = list(node_stats.keys())
        X_list = []
        y_list = []

        feature_names = [
            "in_txns", "out_txns", "total_txns", "in_amount", "out_amount",
            "net_flow", "avg_in_amt", "avg_out_amt", "in_cps_count", "out_cps_count",
            "device_count", "dormancy_days", "age_days", "is_merchant", "is_min_kyc",
            "pagerank", "turnover_ratio", "max_velocity_1h"
        ]

        for vpa in vpas:
            st = node_stats[vpa]
            in_t, out_t = st["in_txns"], st["out_txns"]
            in_a, out_a = st["in_amount"], st["out_amount"]
            
            avg_in = in_a / max(1, in_t)
            avg_out = out_a / max(1, out_t)
            net_flow = in_a - out_a
            turnover_ratio = out_a / max(1.0, in_a) if in_a > 0 else 0.0
            
            # 1-hour burst count
            ts_sorted = sorted(st["timestamps"])
            max_vel = 0
            for idx, t in enumerate(ts_sorted):
                # Count how many within 3600s
                cnt = 0
                for next_t in ts_sorted[idx:]:
                    if next_t - t <= 3600.0:
                        cnt += 1
                    else:
                        break
                max_vel = max(max_vel, cnt)

            feat = [
                in_t,
                out_t,
                in_t + out_t,
                math.log1p(in_a),
                math.log1p(out_a),
                net_flow,
                math.log1p(avg_in),
                math.log1p(avg_out),
                len(st["in_cps"]),
                len(st["out_cps"]),
                len(st["devices"]),
                st["dormancy_days"],
                st["age_days"],
                st["is_merchant"],
                st["is_min_kyc"],
                pagerank.get(vpa, 0.0),
                turnover_ratio,
                max_vel
            ]
            X_list.append(feat)
            y_list.append(1 if vpa in mule_vpas else 0)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), vpas
