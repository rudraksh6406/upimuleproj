"""FastAPI Application Entrypoint for MuleGuard Live Dashboard."""
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import GLOBAL_STATE, router as api_router
from .websocket import ConnectionManager

app = FastAPI(
    title="MuleGuard",
    description="Real-Time UPI Mule Account Detection using Temporal Graph Neural Networks",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Manager
ws_manager = ConnectionManager()
app.include_router(api_router)

# Static files path
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serves the main dashboard single-page application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "MuleGuard API is running. UI assets not found."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transactions, graph, and alert feeds."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection open and accept client messages if any
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
