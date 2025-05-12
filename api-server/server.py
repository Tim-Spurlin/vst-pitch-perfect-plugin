#!/usr/bin/env python3
"""
Real-time vocal transformation API server
Optimized for low-latency, high-quality vocal transformations
"""

import os
import asyncio
import logging
import json
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from websocket_handler import ConnectionManager
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VST Pitch Perfect API",
    description="Real-time vocal transformation API for VST Pitch Perfect Plugin",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize WebSocket connection manager
connection_manager = ConnectionManager()

# Environment variables
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "http://vocal-model-service:8501/v1/models/vocal_transformation_model:predict")
TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "vocal-model-service")
TF_SERVING_PORT = int(os.getenv("TF_SERVING_PORT", "8501"))
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "./models/vocal_transformation_model")

# Initialize audio processor
try:
    if USE_LOCAL_MODEL:
        logger.info(f"Loading local model from {LOCAL_MODEL_PATH}")
        if os.path.exists(LOCAL_MODEL_PATH):
            model = tf.saved_model.load(LOCAL_MODEL_PATH)
            audio_processor = AudioProcessor(local_model=model)
        else:
            logger.warning(f"Local model not found at {LOCAL_MODEL_PATH}. Falling back to TF Serving.")
            audio_processor = AudioProcessor(model_endpoint=MODEL_ENDPOINT)
    else:
        logger.info(f"Using TF Serving model at {MODEL_ENDPOINT}")
        audio_processor = AudioProcessor(model_endpoint=MODEL_ENDPOINT)
except Exception as e:
    logger.error(f"Error initializing audio processor: {e}")
    audio_processor = None

# Performance metrics
request_times = []
request_sizes = []
response_sizes = []

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting VST Pitch Perfect API Server")
    
    # Test connection to model service if using TF Serving
    if not USE_LOCAL_MODEL:
        try:
            import requests
            health_url = f"http://{TF_SERVING_HOST}:{TF_SERVING_PORT}/v1/models/vocal_transformation_model"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to TensorFlow Serving")
            else:
                logger.warning(f"TensorFlow Serving returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to connect to TensorFlow Serving: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down VST Pitch Perfect API Server")
    
    # Log performance metrics
    if request_times:
        avg_request_time = sum(request_times) / len(request_times)
        max_request_time = max(request_times)
        min_request_time = min(request_times)
        logger.info(f"Performance metrics - Avg: {avg_request_time:.2f}s, Min: {min_request_time:.2f}s, Max: {max_request_time:.2f}s")

@app.get("/")
async def read_root():
    """Root endpoint for health checks"""
    return {"status": "Vocal Transformation API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if audio_processor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "Service Unavailable", "message": "Audio processor not initialized"}
        )
    
    # Check model connection
    model_status = "available" if await audio_processor.check_model_connection() else "unavailable"
    
    return {
        "status": "healthy", 
        "model_status": model_status,
        "backend": "local" if USE_LOCAL_MODEL else "tf_serving",
        "metrics": {
            "avg_processing_time": sum(request_times) / len(request_times) if request_times else 0,
            "requests_processed": len(request_times)
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio processing"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Receive audio chunk
            start_time = time.time()
            audio_data = await websocket.receive_bytes()
            
            request_sizes.append(len(audio_data))
            
            if audio_processor is None:
                logger.error("Audio processor not initialized")
                await websocket.send_text(json.dumps({
                    "error": "Service unavailable - Audio processor not initialized"
                }))
                continue
            
            # Process audio (in a non-blocking way)
            try:
                loop = asyncio.get_event_loop()
                transformed_audio = await loop.run_in_executor(
                    None, audio_processor.process_audio_chunk, audio_data)
                
                # Send processed audio back
                await websocket.send_bytes(transformed_audio)
                
                # Track metrics
                response_sizes.append(len(transformed_audio))
                processing_time = time.time() - start_time
                request_times.append(processing_time)
                
                # Log performance periodically
                if len(request_times) % 100 == 0:
                    avg_time = sum(request_times[-100:]) / 100
                    logger.info(f"Average processing time (last 100 requests): {avg_time:.4f}s")
            
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("Client disconnected")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connection_manager.active_connections:
            connection_manager.disconnect(websocket)

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if not request_times:
        return {"message": "No requests processed yet"}
    
    return {
        "requests_processed": len(request_times),
        "avg_processing_time": sum(request_times) / len(request_times),
        "min_processing_time": min(request_times),
        "max_processing_time": max(request_times),
        "avg_request_size_bytes": sum(request_sizes) / len(request_sizes) if request_sizes else 0,
        "avg_response_size_bytes": sum(response_sizes) / len(response_sizes) if response_sizes else 0,
        "active_connections": len(connection_manager.active_connections)
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=port)