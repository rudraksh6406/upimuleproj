"""WebSocket Manager for Live Transaction Streaming & Graph Updates."""
import asyncio
import json
from typing import Any, Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages active WebSocket connections to dashboard clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcasts a structured JSON payload to all connected frontend clients."""
        if not self.active_connections:
            return

        payload = json.dumps({"type": event_type, "data": data})
        dead_sockets = set()
        
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                dead_sockets.add(connection)

        for dead in dead_sockets:
            self.active_connections.discard(dead)
