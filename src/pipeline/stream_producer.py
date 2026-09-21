"""Asynchronous Stream Producer and Consumer for MuleGuard."""
import asyncio
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from .graph_builder import RealTimeGraphBuilder
from .inference import RealTimeInferenceEngine
from ..guardian.decision import GuardianEngine


class AsyncStreamProducer:
    """Simulates live streaming of transactions at configurable playback speeds."""

    def __init__(self, transactions: List[Dict[str, Any]], delay_ms: float = 100.0):
        self.transactions = transactions
        self.delay_ms = delay_ms
        self.is_running = False

    async def stream_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Yields transaction events with async sleep intervals."""
        self.is_running = True
        for txn in self.transactions:
            if not self.is_running:
                break
            yield txn
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000.0)

    def stop(self) -> None:
        self.is_running = False


class StreamProcessor:
    """Coordinates stream intake, real-time graph updates, inference, and Guardian actions."""

    def __init__(
        self,
        inference_engine: RealTimeInferenceEngine,
        graph_builder: RealTimeGraphBuilder,
        guardian: GuardianEngine,
        on_alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.inference_engine = inference_engine
        self.graph_builder = graph_builder
        self.guardian = guardian
        self.on_alert_callback = on_alert_callback

    def process_event(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Full pipeline execution for a single incoming event."""
        # 1. Update Graph
        self.graph_builder.add_transaction(txn)

        # 2. Run TGNN Inference
        inf_result = self.inference_engine.process_transaction(txn)

        # 3. Evaluate Guardian Policies for Sender and Receiver
        decisions = []
        for vpa, score, top_nbrs in [
            (txn["sender_vpa"], inf_result["sender_score"], inf_result["top_sender_neighbors"]),
            (txn["receiver_vpa"], inf_result["receiver_score"], inf_result["top_receiver_neighbors"]),
        ]:
            acc_meta = {
                "vpa": vpa,
                "is_merchant": txn.get("txn_type") == "P2M" and vpa == txn["receiver_vpa"],
                "account_age_days": txn.get("sender_account_age_days", 100) if vpa == txn["sender_vpa"] else txn.get("receiver_account_age_days", 100),
            }
            decision = self.guardian.evaluate(
                account_meta=acc_meta,
                score=score,
                triggering_txn=txn,
                top_neighbors=top_nbrs,
            )
            decisions.append(decision)

            # Update Graph node state
            self.graph_builder.update_node_risk(
                vpa=vpa,
                score=score,
                action=decision["action"],
                explanation=decision["explanation"],
            )

            # Trigger alert callback if critical / warning
            if decision["action"] in ["FREEZE", "RESTRICT", "FLAG"] and self.on_alert_callback:
                self.on_alert_callback(decision)

        return {
            "inference": inf_result,
            "decisions": decisions,
            "transaction": txn,
        }
