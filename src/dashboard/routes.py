"""FastAPI REST API Routes for MuleGuard Dashboard."""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# Global state references initialized by main app
GLOBAL_STATE: Dict[str, Any] = {
    "graph_builder": None,
    "guardian_engine": None,
    "inference_engine": None,
    "metrics_tracker": {
        "total_transactions": 0,
        "total_flagged": 0,
        "total_frozen": 0,
        "latencies": [],
    }
}


class AccountActionRequest(BaseModel):
    account_vpa: str
    action: str  # FREEZE, RESTRICT, UNFREEZE
    reason: str


@router.get("/metrics")
async def get_system_metrics():
    """Returns real-time system metrics (throughput, alerts count, average latency)."""
    tr = GLOBAL_STATE["metrics_tracker"]
    lats = tr["latencies"][-100:]
    avg_lat = round(sum(lats) / max(1, len(lats)), 2) if lats else 12.5
    
    return {
        "total_transactions": tr["total_transactions"],
        "total_flagged": tr["total_flagged"],
        "total_frozen": tr["total_frozen"],
        "avg_latency_ms": avg_lat,
        "system_status": "ONLINE - SCORING LIVE",
    }


@router.get("/graph")
async def get_graph_data(max_nodes: int = Query(80, ge=10, le=200)):
    """Returns dynamic graph snapshot for D3.js visualization."""
    gb = GLOBAL_STATE.get("graph_builder")
    if not gb:
        return {"nodes": [], "links": []}
    return gb.get_subgraph_for_visualization(max_nodes=max_nodes)


@router.get("/alerts")
async def get_recent_alerts(limit: int = Query(25, ge=1, le=100)):
    """Returns recent high-risk alerts from SQLite audit log."""
    guardian = GLOBAL_STATE.get("guardian_engine")
    if not guardian:
        return []
    return guardian.audit_logger.get_recent_alerts(limit=limit)


@router.get("/accounts/{vpa}")
async def get_account_details(vpa: str):
    """Returns detailed history, current risk score, and attention neighbors for an account."""
    gb = GLOBAL_STATE.get("graph_builder")
    if not gb or not gb.G.has_node(vpa):
        raise HTTPException(status_code=404, detail="Account not found in active graph")
        
    data = gb.G.nodes[vpa]
    in_edges = list(gb.G.in_edges(vpa, data=True))
    out_edges = list(gb.G.out_edges(vpa, data=True))

    recent_txns = []
    for u, _, edata in in_edges[-5:]:
        recent_txns.append({
            "direction": "IN",
            "counterpart": u,
            "amount": edata.get("amount", 0),
            "timestamp": edata.get("timestamp", 0),
        })
    for _, v, edata in out_edges[-5:]:
        recent_txns.append({
            "direction": "OUT",
            "counterpart": v,
            "amount": edata.get("amount", 0),
            "timestamp": edata.get("timestamp", 0),
        })

    return {
        "vpa": vpa,
        "risk_score": data.get("risk_score", 0.0),
        "action": data.get("action", "ALLOW"),
        "explanation": data.get("explanation", "Normal account"),
        "is_merchant": data.get("is_merchant", False),
        "in_degree": gb.G.in_degree(vpa),
        "out_degree": gb.G.out_degree(vpa),
        "recent_transactions": recent_txns,
    }


@router.post("/accounts/action")
async def override_account_action(req: AccountActionRequest):
    """Allows human risk officer to manually freeze/restrict/unfreeze an account."""
    gb = GLOBAL_STATE.get("graph_builder")
    guardian = GLOBAL_STATE.get("guardian_engine")
    
    if gb and gb.G.has_node(req.account_vpa):
        gb.update_node_risk(
            vpa=req.account_vpa,
            score=1.0 if req.action == "FREEZE" else 0.0,
            action=req.action,
            explanation=f"Manual operator action: {req.reason}",
        )
    return {"status": "success", "account": req.account_vpa, "action": req.action}
