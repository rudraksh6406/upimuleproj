"""Mule Fraud Pattern Injectors for Synthetic UPI Transaction Streams.

Implements all 9 distinct mule topology & temporal patterns described in PRD Section 6.2:
1. Fan-out
2. Layering Chain
3. Fan-in Aggregator
4. Cyclic Flow
5. Burst-Dormant
6. Coordinated Mule Ring
7. Device-Sharing Cluster
8. Amount-Clustering Attack
9. Gaming/Betting Funnel
"""
import random
import time
import uuid
from typing import Any, Dict, List, Set, Tuple


def _create_txn_event(
    txn_id: str,
    sender: Dict[str, Any],
    receiver: Dict[str, Any],
    amount: float,
    timestamp: float,
    txn_type: str = "P2P",
    device_id: str = None,
    is_mule_sender: bool = False,
    is_mule_receiver: bool = False,
    pattern_type: str = None,
    merchant_category: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Helper to build a standardized transaction event dict."""
    dev_id = device_id or sender.get("primary_device", "dev_default")
    return {
        "txn_id": txn_id,
        "sender_vpa": sender["vpa"],
        "receiver_vpa": receiver["vpa"],
        "amount": round(float(amount), 2),
        "currency": "INR",
        "timestamp": float(timestamp),
        "txn_type": txn_type,
        "device_id": dev_id,
        "device_type": sender.get("device_type", "android"),
        "merchant_category": merchant_category,
        "sender_kyc_type": sender.get("kyc_type", "full"),
        "receiver_kyc_type": receiver.get("kyc_type", "full"),
        "sender_account_age_days": sender.get("account_age_days", 100),
        "receiver_account_age_days": receiver.get("account_age_days", 100),
        "is_mule_sender": is_mule_sender,
        "is_mule_receiver": is_mule_receiver,
        "mule_pattern_type": pattern_type,
        "metadata": metadata or {
            "geo_city": sender.get("city", "Bengaluru"),
            "geo_state": sender.get("state", "Karnataka"),
            "app_name": sender.get("app_name", "PhonePe"),
            "os_version": "Android 14"
        }
    }


def inject_fan_out(
    source_account: Dict[str, Any],
    mule_accounts: List[Dict[str, Any]],
    exit_accounts: List[Dict[str, Any]],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 1: Fan-Out Mule Network.
    1 source sends to 5-20 mules within 10 minutes.
    Each mule forwards onward after a short delay.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(mule_accounts), random.randint(5, 15))
    selected_mules = mule_accounts[:num_mules]
    
    # Step 1: Source fans out to mules within 10 min (600s)
    current_time = start_time
    for i, mule in enumerate(selected_mules):
        mule_vpa = mule["vpa"]
        mules_identified.add(mule_vpa)
        mule["is_mule"] = True
        mule["mule_pattern"] = "fan_out"
        
        amount = random.uniform(5000.0, 45000.0)
        current_time += random.uniform(10.0, 45.0)
        txn = _create_txn_event(
            txn_id=f"txn_fanout_in_{int(current_time)}_{i}",
            sender=source_account,
            receiver=mule,
            amount=amount,
            timestamp=current_time,
            is_mule_sender=False,  # source might be compromised legit account
            is_mule_receiver=True,
            pattern_type="fan_out",
        )
        txns.append(txn)
        
        # Step 2: Mule forwards onward after 2-8 minutes
        forward_time = current_time + random.uniform(120.0, 480.0)
        forward_amount = amount * random.uniform(0.90, 0.95)
        exit_acc = random.choice(exit_accounts)
        fwd_txn = _create_txn_event(
            txn_id=f"txn_fanout_out_{int(forward_time)}_{i}",
            sender=mule,
            receiver=exit_acc,
            amount=forward_amount,
            timestamp=forward_time,
            is_mule_sender=True,
            is_mule_receiver=False,
            pattern_type="fan_out",
        )
        txns.append(fwd_txn)
        
    return txns, mules_identified


def inject_layering_chain(
    chain_mules: List[Dict[str, Any]],
    start_time: float,
    initial_amount: float = 50000.0,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 2: Layering Chain.
    A -> B -> C -> D (3-5 hops).
    Each hop occurs within 2-10 minutes with 5-10% commission deduction.
    """
    txns = []
    mules_identified = set()
    chain_length = min(len(chain_mules), random.randint(3, 5))
    selected = chain_mules[:chain_length]
    
    current_time = start_time
    current_amount = initial_amount
    fraud_device = f"dev_layering_{random.randint(1000, 9999)}"
    
    for i in range(chain_length - 1):
        sender = selected[i]
        receiver = selected[i + 1]
        
        sender["is_mule"] = True
        sender["mule_pattern"] = "layering_chain"
        receiver["is_mule"] = True
        receiver["mule_pattern"] = "layering_chain"
        mules_identified.add(sender["vpa"])
        mules_identified.add(receiver["vpa"])
        
        current_time += random.uniform(120.0, 600.0)  # 2 to 10 minutes
        current_amount *= random.uniform(0.90, 0.95)  # 5-10% cut
        
        txn = _create_txn_event(
            txn_id=f"txn_layering_{int(current_time)}_{i}",
            sender=sender,
            receiver=receiver,
            amount=current_amount,
            timestamp=current_time,
            device_id=fraud_device if random.random() < 0.6 else None,
            is_mule_sender=True,
            is_mule_receiver=True,
            pattern_type="layering_chain",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_fan_in(
    mule_senders: List[Dict[str, Any]],
    aggregator: Dict[str, Any],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 3: Fan-In Aggregator.
    10-30 mules send to 1 aggregator spread over 30-60 minutes.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(mule_senders), random.randint(8, 20))
    selected_mules = mule_senders[:num_mules]
    
    aggregator["is_mule"] = True
    aggregator["mule_pattern"] = "fan_in"
    mules_identified.add(aggregator["vpa"])
    
    current_time = start_time
    for i, sender in enumerate(selected_mules):
        sender["is_mule"] = True
        sender["mule_pattern"] = "fan_in"
        mules_identified.add(sender["vpa"])
        
        current_time += random.uniform(60.0, 200.0)  # spread across 30-60 mins
        amount = random.uniform(3000.0, 25000.0)
        
        txn = _create_txn_event(
            txn_id=f"txn_fanin_{int(current_time)}_{i}",
            sender=sender,
            receiver=aggregator,
            amount=amount,
            timestamp=current_time,
            is_mule_sender=True,
            is_mule_receiver=True,
            pattern_type="fan_in",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_cyclic_flow(
    cycle_nodes: List[Dict[str, Any]],
    start_time: float,
    amount: float = 30000.0,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 4: Cyclic Flow (Wash-Trading / Circular Laundering).
    A -> B -> C -> A within 1 hour.
    """
    txns = []
    mules_identified = set()
    cycle_len = min(len(cycle_nodes), random.randint(3, 5))
    selected = cycle_nodes[:cycle_len]
    
    current_time = start_time
    for i in range(cycle_len):
        sender = selected[i]
        receiver = selected[(i + 1) % cycle_len]
        
        sender["is_mule"] = True
        sender["mule_pattern"] = "cyclic_flow"
        mules_identified.add(sender["vpa"])
        
        current_time += random.uniform(300.0, 900.0)  # 5-15 mins per hop
        txn_amount = amount * random.uniform(0.97, 1.03)  # minor fluctuation
        
        txn = _create_txn_event(
            txn_id=f"txn_cycle_{int(current_time)}_{i}",
            sender=sender,
            receiver=receiver,
            amount=txn_amount,
            timestamp=current_time,
            is_mule_sender=True,
            is_mule_receiver=True,
            pattern_type="cyclic_flow",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_burst_dormant(
    dormant_account: Dict[str, Any],
    counterparts: List[Dict[str, Any]],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 5: Burst-Dormant.
    Account was dormant for 60+ days, suddenly executes 5-15 txns in 30 mins, then goes dormant.
    """
    txns = []
    mules_identified = {dormant_account["vpa"]}
    dormant_account["is_mule"] = True
    dormant_account["mule_pattern"] = "burst_dormant"
    dormant_account["dormancy_days"] = random.uniform(60.0, 180.0)
    
    burst_count = random.randint(6, 14)
    current_time = start_time
    
    for i in range(burst_count):
        current_time += random.uniform(60.0, 150.0)  # within 30 min window
        cp = random.choice(counterparts)
        is_incoming = (i % 2 == 0)
        
        sender = cp if is_incoming else dormant_account
        receiver = dormant_account if is_incoming else cp
        
        txn = _create_txn_event(
            txn_id=f"txn_burstdormant_{int(current_time)}_{i}",
            sender=sender,
            receiver=receiver,
            amount=random.uniform(4000.0, 35000.0),
            timestamp=current_time,
            is_mule_sender=(sender["vpa"] == dormant_account["vpa"]),
            is_mule_receiver=(receiver["vpa"] == dormant_account["vpa"]),
            pattern_type="burst_dormant",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_coordinated_ring(
    ring_mules: List[Dict[str, Any]],
    source_account: Dict[str, Any],
    destination_account: Dict[str, Any],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 6: Coordinated Mule Ring.
    5-20 mules activate within the same hour, receive from same source, forward to same destination.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(ring_mules), random.randint(5, 12))
    selected = ring_mules[:num_mules]
    
    for i, mule in enumerate(selected):
        mule["is_mule"] = True
        mule["mule_pattern"] = "coordinated_ring"
        mules_identified.add(mule["vpa"])
        
        # Inflow from common source
        t_in = start_time + random.uniform(0.0, 1800.0)
        amount = random.uniform(8000.0, 40000.0)
        txn_in = _create_txn_event(
            txn_id=f"txn_coord_in_{int(t_in)}_{i}",
            sender=source_account,
            receiver=mule,
            amount=amount,
            timestamp=t_in,
            is_mule_sender=False,
            is_mule_receiver=True,
            pattern_type="coordinated_ring",
        )
        txns.append(txn_in)
        
        # Outflow to common destination within 1 hour
        t_out = t_in + random.uniform(300.0, 1200.0)
        txn_out = _create_txn_event(
            txn_id=f"txn_coord_out_{int(t_out)}_{i}",
            sender=mule,
            receiver=destination_account,
            amount=amount * random.uniform(0.92, 0.98),
            timestamp=t_out,
            is_mule_sender=True,
            is_mule_receiver=False,
            pattern_type="coordinated_ring",
        )
        txns.append(txn_out)
        
    return txns, mules_identified


def inject_device_sharing(
    shared_mules: List[Dict[str, Any]],
    recipients: List[Dict[str, Any]],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 7: Device-Sharing Mule Cluster.
    10-30 accounts all operated from the same single device fingerprint.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(shared_mules), random.randint(10, 25))
    selected = shared_mules[:num_mules]
    shared_device_id = f"dev_shared_ring_{random.randint(10000, 99999)}"
    
    current_time = start_time
    for i, mule in enumerate(selected):
        mule["is_mule"] = True
        mule["mule_pattern"] = "device_sharing"
        mule["primary_device"] = shared_device_id
        mules_identified.add(mule["vpa"])
        
        current_time += random.uniform(100.0, 500.0)
        rec = random.choice(recipients)
        txn = _create_txn_event(
            txn_id=f"txn_devshare_{int(current_time)}_{i}",
            sender=mule,
            receiver=rec,
            amount=random.uniform(2000.0, 18000.0),
            timestamp=current_time,
            device_id=shared_device_id,
            is_mule_sender=True,
            is_mule_receiver=False,
            pattern_type="device_sharing",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_amount_clustering(
    bot_mules: List[Dict[str, Any]],
    merchants: List[Dict[str, Any]],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 8: Amount-Clustering Attack (Bot-driven card testing / micro-bursts).
    Accounts transacting at near-identical amounts (e.g. ₹211 or ₹262) across merchants.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(bot_mules), random.randint(15, 30))
    selected = bot_mules[:num_mules]
    cluster_amount = random.choice([211.0, 262.0, 499.0, 101.0])
    
    current_time = start_time
    for i, mule in enumerate(selected):
        mule["is_mule"] = True
        mule["mule_pattern"] = "amount_clustering"
        mules_identified.add(mule["vpa"])
        
        current_time += random.uniform(30.0, 120.0)
        merchant = random.choice(merchants)
        txn = _create_txn_event(
            txn_id=f"txn_amtcluster_{int(current_time)}_{i}",
            sender=mule,
            receiver=merchant,
            amount=cluster_amount,
            timestamp=current_time,
            txn_type="P2M",
            merchant_category=merchant.get("merchant_category", "shopping"),
            is_mule_sender=True,
            is_mule_receiver=False,
            pattern_type="amount_clustering",
        )
        txns.append(txn)
        
    return txns, mules_identified


def inject_gaming_betting(
    funnel_mules: List[Dict[str, Any]],
    payers: List[Dict[str, Any]],
    start_time: float,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Pattern 9: Gaming/Betting Funnel.
    Accounts categorized under disguised merchant labels processing high-velocity funnel payments.
    """
    txns = []
    mules_identified = set()
    num_mules = min(len(funnel_mules), random.randint(4, 10))
    selected = funnel_mules[:num_mules]
    
    current_time = start_time
    for i, mule in enumerate(selected):
        mule["is_mule"] = True
        mule["mule_pattern"] = "gaming_betting"
        mules_identified.add(mule["vpa"])
        
        # High volume of payments from multiple casual payers
        for p_idx in range(random.randint(5, 12)):
            current_time += random.uniform(40.0, 180.0)
            payer = random.choice(payers)
            txn = _create_txn_event(
                txn_id=f"txn_gaming_{int(current_time)}_{i}_{p_idx}",
                sender=payer,
                receiver=mule,
                amount=random.uniform(500.0, 15000.0),
                timestamp=current_time,
                txn_type="P2M",
                merchant_category="entertainment",
                is_mule_sender=False,
                is_mule_receiver=True,
                pattern_type="gaming_betting",
            )
            txns.append(txn)
            
    return txns, mules_identified
