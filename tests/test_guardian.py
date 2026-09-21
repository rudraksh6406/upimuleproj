"""Unit tests for Guardian decision engine and real-time inference pipeline."""
import os
import pytest
from src.guardian.decision import GuardianEngine, GuardianAuditLogger
from src.pipeline.graph_builder import RealTimeGraphBuilder
from src.pipeline.inference import RealTimeInferenceEngine
from src.models.tgn import TGNModel
from src.data.preprocessor import GraphPreprocessor


def test_guardian_decision_thresholds():
    test_db = "test_audit.db"
    try:
        guardian = GuardianEngine(db_path=test_db)
        
        # Test Freeze threshold
        meta = {"vpa": "mule@okaxis", "account_age_days": 100, "is_merchant": False}
        txn = {"txn_id": "t1", "timestamp": 1000.0, "amount": 50000.0, "mule_pattern_type": "layering_chain"}
        
        dec_freeze = guardian.evaluate(meta, score=0.95, triggering_txn=txn)
        assert dec_freeze["action"] == "FREEZE"
        assert "layering" in dec_freeze["explanation"].lower()

        # Test Allow threshold
        dec_allow = guardian.evaluate(meta, score=0.10, triggering_txn=txn)
        assert dec_allow["action"] == "ALLOW"
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_realtime_pipeline():
    prep = GraphPreprocessor()
    model = TGNModel(num_nodes=20)
    inf_engine = RealTimeInferenceEngine(model=model, preprocessor=prep)
    gb = RealTimeGraphBuilder()

    txn = {
        "txn_id": "txn_test_001",
        "sender_vpa": "alice@okaxis",
        "receiver_vpa": "bob@okhdfcbank",
        "amount": 1500.0,
        "timestamp": 1726910400.0,
        "txn_type": "P2P",
        "device_id": "dev_001",
    }

    gb.add_transaction(txn)
    assert gb.G.has_edge("alice@okaxis", "bob@okhdfcbank")

    res = inf_engine.process_transaction(txn)
    assert res["sender_score"] >= 0.0
    assert res["receiver_score"] >= 0.0
    assert res["latency_ms"] < 100.0  # Latency check
