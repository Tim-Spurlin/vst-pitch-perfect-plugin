#!/usr/bin/env python3
"""
WebSocket connection handler for real-time audio processing
Optimized for low-latency, high-quality vocal transformations
"""

import asyncio
import logging
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    WebSocket connection manager for handling multiple client connections
    Optimized for real-time audio processing
    """
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        self.stats = {
            "total_connections": 0,
            "max_concurrent_connections": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "bytes_received": 0,
            "bytes_sent": 0
        }
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Initialize connection info
        self.connection_info[websocket] = {
            "id": len(self.active_connections),
            "connected_at": asyncio.get_event_loop().time(),
            "messages_received": 0,
            "messages_sent": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
            "last_message_at": asyncio.get_event_loop().time()
        }
        
        # Update stats
        self.stats["total_connections"] += 1
        self.stats["max_concurrent_connections"] = max(
            self.stats["max_concurrent_connections"],
            len(self.active_connections)
        )
        
        logger.info(f"New connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Cleanup connection info
            if websocket in self.connection_info:
                connection_duration = asyncio.get_event_loop().time() - self.connection_info[websocket]["connected_at"]
                logger.info(f"Connection {self.connection_info[websocket]['id']} disconnected after {connection_duration:.2f}s")
                del self.connection_info[websocket]
            
            logger.info(f"Connection closed. Remaining connections: {len(self.active_connections)}")
    
    async def send_text(self, websocket: WebSocket, message: str):
        """Send text message to a WebSocket client with tracking"""
        try:
            await websocket.send_text(message)
            
            # Update stats
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message.encode('utf-8'))
            
            # Update connection info
            if websocket in self.connection_info:
                self.connection_info[websocket]["messages_sent"] += 1
                self.connection_info[websocket]["bytes_sent"] += len(message.encode('utf-8'))
                self.connection_info[websocket]["last_message_at"] = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error(f"Error sending text to WebSocket: {e}")
            # Disconnect client on error
            self.disconnect(websocket)
    
    async def send_bytes(self, websocket: WebSocket, data: bytes):
        """Send binary data to a WebSocket client with tracking"""
        try:
            await websocket.send_bytes(data)
            
            # Update stats
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(data)
            
            # Update connection info
            if websocket in self.connection_info:
                self.connection_info[websocket]["messages_sent"] += 1
                self.connection_info[websocket]["bytes_sent"] += len(data)
                self.connection_info[websocket]["last_message_at"] = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error(f"Error sending bytes to WebSocket: {e}")
            # Disconnect client on error
            self.disconnect(websocket)
    
    async def broadcast_text(self, message: str):
        """Broadcast text message to all connected clients"""
        for connection in self.active_connections:
            await self.send_text(connection, message)
    
    async def broadcast_bytes(self, data: bytes):
        """Broadcast binary data to all connected clients"""
        for connection in self.active_connections:
            await self.send_bytes(connection, data)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected clients"""
        json_str = json.dumps(data)
        await self.broadcast_text(json_str)
    
    async def check_inactive_connections(self, timeout: float = 60.0):
        """Check for and close inactive connections"""
        current_time = asyncio.get_event_loop().time()
        
        inactive_connections = []
        for ws, info in self.connection_info.items():
            inactive_time = current_time - info["last_message_at"]
            if inactive_time > timeout:
                inactive_connections.append(ws)
        
        for ws in inactive_connections:
            logger.info(f"Closing inactive connection {self.connection_info[ws]['id']} "
                      f"(inactive for {current_time - self.connection_info[ws]['last_message_at']:.2f}s)")
            try:
                await ws.close(code=1000, reason="Inactive connection")
            except Exception:
                pass
            self.disconnect(ws)
    
    def track_received_message(self, websocket: WebSocket, size: int):
        """Track received message statistics"""
        # Update stats
        self.stats["messages_received"] += 1
        self.stats["bytes_received"] += size
        
        # Update connection info
        if websocket in self.connection_info:
            self.connection_info[websocket]["messages_received"] += 1
            self.connection_info[websocket]["bytes_received"] += size
            self.connection_info[websocket]["last_message_at"] = asyncio.get_event_loop().time()
    
    def get_connection_stats(self):
        """Get connection statistics"""
        active_connections = []
        for ws, info in self.connection_info.items():
            connection_duration = asyncio.get_event_loop().time() - info["connected_at"]
            active_connections.append({
                "id": info["id"],
                "connection_duration": connection_duration,
                "messages_received": info["messages_received"],
                "messages_sent": info["messages_sent"],
                "bytes_received": info["bytes_received"],
                "bytes_sent": info["bytes_sent"],
                "last_activity": asyncio.get_event_loop().time() - info["last_message_at"]
            })
        
        return {
            "stats": self.stats,
            "active_connections": active_connections
        }

async def inactive_connection_checker(connection_manager: ConnectionManager, check_interval: float = 30.0):
    """Background task to check for inactive connections"""
    while True:
        await asyncio.sleep(check_interval)
        await connection_manager.check_inactive_connections()

def start_background_tasks(connection_manager: ConnectionManager):
    """Start background tasks for the connection manager"""
    loop = asyncio.get_event_loop()
    loop.create_task(inactive_connection_checker(connection_manager))