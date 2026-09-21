"""Guardian Decision Engine, Policy Thresholds, Actions & SQLite Audit Log."""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .explanation import generate_explanation


class GuardianAuditLogger:
    """Persistent SQLite database logger for compliance, action history, and examiner demos."""

    def __init__(self, db_path: str = "muleguard.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    account_vpa TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    mule_score REAL NOT NULL,
                    action TEXT NOT NULL,
                    triggering_txn_id TEXT,
                    explanation TEXT,
                    metadata_json TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_vpa TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    alert_level TEXT NOT NULL,
                    action TEXT NOT NULL,
                    score REAL NOT NULL,
                    explanation TEXT
                )
            """)
            conn.commit()

    def log_decision(self, decision: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO decisions 
                (decision_id, account_vpa, timestamp, mule_score, action, triggering_txn_id, explanation, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision["decision_id"],
                decision["account_id"],
                decision["timestamp"],
                decision["mule_score"],
                decision["action"],
                decision.get("triggering_txn_id", ""),
                decision.get("explanation", ""),
                json.dumps(decision.get("top_5_attention_neighbors", []))
            ))
            
            if decision["action"] in ["FREEZE", "RESTRICT", "FLAG"]:
                level = "critical" if decision["action"] == "FREEZE" else ("warning" if decision["action"] == "RESTRICT" else "info")
                cursor.execute("""
                    INSERT INTO alerts (account_vpa, timestamp, alert_level, action, score, explanation)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    decision["account_id"],
                    decision["timestamp"],
                    level,
                    decision["action"],
                    decision["mule_score"],
                    decision["explanation"]
                ))
            conn.commit()

    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]


class GuardianEngine:
    """Automated Real-Time Decision & Action Policy Engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, db_path: str = "muleguard.db"):
        self.config = config or {}
        self.thresholds = self.config.get("thresholds", {
            "normal_account": {"flag": 0.30, "restrict": 0.70, "freeze": 0.90},
            "new_account": {"flag": 0.20, "restrict": 0.50, "freeze": 0.80},
            "merchant_account": {"flag": 0.30, "restrict": 0.70, "freeze": 0.92},
        })
        self.actions_cfg = self.config.get("actions", {
            "freeze": {"block_incoming": True, "block_outgoing": True, "alert_level": "critical"},
            "restrict": {"block_incoming": False, "block_outgoing": True, "alert_level": "warning"},
            "flag": {"block_incoming": False, "block_outgoing": False, "alert_level": "info"},
        })
        self.audit_logger = GuardianAuditLogger(db_path=db_path)
        self.decision_counter = 0

    def get_applicable_thresholds(self, account_meta: Dict[str, Any]) -> Dict[str, float]:
        """Selects tailored threshold configuration based on account type and tenure."""
        if account_meta.get("is_merchant"):
            return self.thresholds.get("merchant_account", {"flag": 0.3, "restrict": 0.7, "freeze": 0.92})
        elif account_meta.get("account_age_days", 100) < 7:
            return self.thresholds.get("new_account", {"flag": 0.2, "restrict": 0.5, "freeze": 0.80})
        else:
            return self.thresholds.get("normal_account", {"flag": 0.3, "restrict": 0.7, "freeze": 0.90})

    def evaluate(
        self,
        account_meta: Dict[str, Any],
        score: float,
        triggering_txn: Dict[str, Any],
        top_neighbors: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluates probability score against risk policy, assigns action, logs, and produces output matching PRD Appendix D.3."""
        self.decision_counter += 1
        t = float(triggering_txn.get("timestamp", time.time()))
        vpa = account_meta["vpa"]
        thresh = self.get_applicable_thresholds(account_meta)

        # Policy Action Mapping
        if score >= thresh["freeze"]:
            action = "FREEZE"
        elif score >= thresh["restrict"]:
            action = "RESTRICT"
        elif score >= thresh["flag"]:
            action = "FLAG"
        else:
            action = "ALLOW"

        # Generate Human-Readable Explanation
        top_nbrs = top_neighbors or []
        explanation = generate_explanation(
            account_vpa=vpa,
            score=score,
            action=action,
            triggering_txn=triggering_txn,
            top_neighbors=top_nbrs,
            account_meta=account_meta,
        )

        decision = {
            "decision_id": f"dec_{int(t)}_{self.decision_counter:06d}",
            "account_id": vpa,
            "timestamp": t,
            "mule_score": round(float(score), 4),
            "action": action,
            "thresholds_used": thresh,
            "triggering_txn_id": triggering_txn.get("txn_id", ""),
            "top_5_attention_neighbors": top_nbrs,
            "explanation": explanation,
            "human_review_status": "pending" if action in ["FREEZE", "RESTRICT", "FLAG"] else "none"
        }

        # Save to SQLite Audit Log
        self.audit_logger.log_decision(decision)

        return decision
