"""Natural Language Explanation Generator for Guardian Risk Engine."""
from typing import Any, Dict, List, Optional


def generate_explanation(
    account_vpa: str,
    score: float,
    action: str,
    triggering_txn: Dict[str, Any],
    top_neighbors: List[Dict[str, Any]],
    account_meta: Dict[str, Any],
) -> str:
    """Generates human-readable, auditable explanation for risk officers based on attention weights and flow signals."""
    if action == "ALLOW":
        return f"Account {account_vpa} scored {score:.2f} (Normal) — within legitimate behavioral parameters."

    reasons = []
    
    # 1. Pattern and Amount signals
    pattern = triggering_txn.get("mule_pattern_type")
    amt = float(triggering_txn.get("amount", 0.0))
    
    if pattern == "layering_chain":
        reasons.append(f"Rapid fund pass-through / layering detected (₹{amt:,.2f}) forwarded within minutes.")
    elif pattern == "fan_out":
        reasons.append(f"High-frequency dispersal / fan-out to multiple downstream accounts.")
    elif pattern == "fan_in":
        reasons.append(f"High-volume aggregation / fan-in from multiple distinct senders.")
    elif pattern == "burst_dormant":
        reasons.append(f"Sudden high-velocity transaction burst following extended dormancy period.")
    elif pattern == "device_sharing":
        reasons.append(f"Shared device fingerprint associated with coordinated fraud ring.")
    elif pattern == "cyclic_flow":
        reasons.append(f"Circular fund routing (wash-trade / cycle) returning funds to origin.")
    elif pattern == "amount_clustering":
        reasons.append(f"Repeated identical transaction amounts characteristic of bot-driven card testing.")
    elif pattern == "gaming_betting":
        reasons.append(f"Unregistered high-velocity gaming/betting merchant funnel activity.")
    else:
        if amt > 25000:
            reasons.append(f"Unusual high-value transfer of ₹{amt:,.2f}.")
        else:
            reasons.append(f"Anomalous temporal flow sequence and rapid turnover.")

    # 2. Attention neighbor influence
    if top_neighbors:
        top_nbr = top_neighbors[0]
        nbr_vpa = top_nbr.get("vpa", "unknown")
        weight = top_nbr.get("attention_weight", 0.0)
        reasons.append(f"High temporal graph attention weight ({weight:.2f}) with connected node '{nbr_vpa}'.")

    # 3. Account age context
    age = account_meta.get("account_age_days", 100)
    if age < 7:
        reasons.append(f"New account ({age} days old) exhibiting immediate high velocity.")

    details = " ".join(reasons)
    return f"Account {account_vpa} {action} (Score: {score:.2f}): {details}"
