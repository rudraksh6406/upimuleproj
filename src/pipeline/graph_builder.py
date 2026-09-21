"""Dynamic In-Memory Temporal Graph Builder for Real-Time Streaming."""
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import networkx as nx


class RealTimeGraphBuilder:
    """Maintains a rolling temporal transaction graph in memory with fast edge and neighbor queries."""

    def __init__(self, retention_window_seconds: float = 86400.0, max_recent_nodes: int = 150):
        self.retention_seconds = retention_window_seconds
        self.max_recent_nodes = max_recent_nodes
        
        # In-memory dynamic graph
        self.G = nx.DiGraph()
        self.recent_txns: Deque[Dict[str, Any]] = deque()
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self.active_vpas: Deque[str] = deque(maxlen=max_recent_nodes)

    def add_transaction(self, txn: Dict[str, Any]) -> None:
        """Adds incoming transaction to graph, updates node metadata, and purges expired edges."""
        u = txn["sender_vpa"]
        v = txn["receiver_vpa"]
        amt = float(txn.get("amount", 0.0))
        ts = float(txn.get("timestamp", time.time()))
        
        self.recent_txns.append(txn)
        if u not in self.active_vpas:
            self.active_vpas.append(u)
        if v not in self.active_vpas:
            self.active_vpas.append(v)

        # Update NetworkX Graph
        if not self.G.has_node(u):
            self.G.add_node(u, vpa=u, is_merchant=txn.get("txn_type") == "P2M", risk_score=0.0, action="ALLOW")
        if not self.G.has_node(v):
            self.G.add_node(v, vpa=v, is_merchant=txn.get("txn_type") == "P2M", risk_score=0.0, action="ALLOW")

        self.G.add_edge(u, v, amount=amt, timestamp=ts, txn_id=txn.get("txn_id"))

        # Sliding window pruning
        cutoff = ts - self.retention_seconds
        while self.recent_txns and self.recent_txns[0]["timestamp"] < cutoff:
            expired = self.recent_txns.popleft()
            exp_u, exp_v = expired["sender_vpa"], expired["receiver_vpa"]
            if self.G.has_edge(exp_u, exp_v):
                # Only remove if this was the last edge
                if self.G[exp_u][exp_v].get("timestamp", 0) <= expired["timestamp"]:
                    self.G.remove_edge(exp_u, exp_v)

    def update_node_risk(self, vpa: str, score: float, action: str, explanation: str = "") -> None:
        """Updates real-time score and action state on node."""
        if self.G.has_node(vpa):
            self.G.nodes[vpa]["risk_score"] = round(float(score), 4)
            self.G.nodes[vpa]["action"] = action
            self.G.nodes[vpa]["explanation"] = explanation

    def get_subgraph_for_visualization(self, max_nodes: int = 80) -> Dict[str, Any]:
        """Returns JSON-serializable node and link structure for D3.js force graph."""
        nodes_to_include = list(self.active_vpas)[-max_nodes:] if self.active_vpas else list(self.G.nodes())[:max_nodes]
        subgraph = self.G.subgraph(nodes_to_include)

        node_list = []
        for n in subgraph.nodes():
            data = self.G.nodes[n]
            node_list.append({
                "id": n,
                "vpa": n,
                "is_merchant": data.get("is_merchant", False),
                "risk_score": data.get("risk_score", 0.0),
                "action": data.get("action", "ALLOW"),
                "in_degree": subgraph.in_degree(n),
                "out_degree": subgraph.out_degree(n),
            })

        link_list = []
        for u, v, data in subgraph.edges(data=True):
            link_list.append({
                "source": u,
                "target": v,
                "amount": data.get("amount", 0.0),
                "timestamp": data.get("timestamp", 0.0),
                "txn_id": data.get("txn_id", ""),
            })

        return {"nodes": node_list, "links": link_list}
