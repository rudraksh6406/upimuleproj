#!/usr/bin/env python3
"""One-Command Live Demonstration Runner for MuleGuard (Stream + Inference + Guardian + Web UI)."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import uvicorn

from src.dashboard.app import app, ws_manager
from src.dashboard.routes import GLOBAL_STATE
from src.data.generator import UPIDataGenerator
from src.data.preprocessor import GraphPreprocessor
from src.guardian.decision import GuardianEngine
from src.models.tgn import TGNModel
from src.pipeline.graph_builder import RealTimeGraphBuilder
from src.pipeline.inference import RealTimeInferenceEngine
from src.pipeline.stream_producer import AsyncStreamProducer, StreamProcessor
from src.utils.config import load_all_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("run_demo")


async def streaming_worker(processor: StreamProcessor, producer: AsyncStreamProducer):
    """Background task streaming transactions and broadcasting to WebSockets."""
    logger.info("Starting live transaction playback stream...")
    
    async for txn in producer.stream_events():
        # Process event through TGNN inference + Guardian
        result = processor.process_event(txn)
        
        # Update metrics tracker
        tr = GLOBAL_STATE["metrics_tracker"]
        tr["total_transactions"] += 1
        tr["latencies"].append(result["inference"]["latency_ms"])

        for dec in result["decisions"]:
            if dec["action"] in ["FREEZE", "RESTRICT", "FLAG"]:
                tr["total_flagged"] += 1
                if dec["action"] == "FREEZE":
                    tr["total_frozen"] += 1
                # Broadcast alert to WebSockets
                await ws_manager.broadcast_event("ALERT", dec)

        # Periodically broadcast updated graph
        if tr["total_transactions"] % 2 == 0:
            graph_snapshot = processor.graph_builder.get_subgraph_for_visualization(max_nodes=80)
            await ws_manager.broadcast_event("GRAPH_UPDATE", graph_snapshot)


@app.on_event("startup")
async def on_startup():
    """Initializes demo components and spawns streaming worker."""
    set_seed(42)
    configs = load_all_configs()
    
    # 1. Generate / Load Demo Stream Data
    logger.info("Generating demo transaction stream with planted mule fraud rings...")
    data_gen = UPIDataGenerator(config=configs["data"])
    demo_txns, demo_accs, demo_mules = data_gen.generate_dataset(
        num_accounts=300, time_span_days=3, base_timestamp=1726910400.0
    )
    logger.info(f"Stream ready: {len(demo_txns)} transactions, {len(demo_mules)} planted mule accounts.")

    # 2. Initialize Models & Engines
    prep = GraphPreprocessor()
    # Pre-register nodes
    for acc in demo_accs:
        prep.get_or_create_node_id(acc["vpa"])

    model = TGNModel(num_nodes=len(demo_accs) + 50)
    
    # Load weights if available
    ckpt_path = root_dir / "results" / "checkpoints" / "best_tgn.pt"
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info("Loaded trained TGN model weights into demo engine.")
        except Exception:
            pass

    graph_builder = RealTimeGraphBuilder(retention_window_seconds=86400.0, max_recent_nodes=100)
    guardian = GuardianEngine(config=configs["guardian"])
    inf_engine = RealTimeInferenceEngine(model=model, preprocessor=prep)

    # Set Global references for REST API
    GLOBAL_STATE["graph_builder"] = graph_builder
    GLOBAL_STATE["guardian_engine"] = guardian
    GLOBAL_STATE["inference_engine"] = inf_engine

    processor = StreamProcessor(
        inference_engine=inf_engine,
        graph_builder=graph_builder,
        guardian=guardian,
    )

    producer = AsyncStreamProducer(transactions=demo_txns, delay_ms=180.0)

    # Spawn background task
    asyncio.create_task(streaming_worker(processor, producer))


def main():
    parser = argparse.ArgumentParser(description="Start MuleGuard Live Demo Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("\n" + "="*70)
    print(" 🛡️  MULEGUARD — REAL-TIME UPI MULE DETECTION DEMO")
    print("="*70)
    print(f" Dashboard URL: http://{args.host}:{args.port}")
    print(" REST API Docs: http://127.0.0.1:8000/docs")
    print(" Press Ctrl+C to terminate.")
    print("="*70 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
