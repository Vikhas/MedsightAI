"""
MediSight — FastAPI Backend Server

Serves the frontend static files and provides a WebSocket endpoint
that proxies audio/video between the browser and Gemini Live API.
"""

import asyncio
import base64
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from gemini_live import GeminiLiveSession

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# FastAPI app
app = FastAPI(
    title="MediSight — Real-Time Clinical Decision Assistant",
    description="Gemini Live Agent Challenge Hackathon Project",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "MediSight Backend",
        "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY")),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint that bridges the browser and Gemini Live API.

    Messages from browser → Gemini:
        {"type": "audio", "data": "<base64 PCM>"}
        {"type": "video", "data": "<base64 JPEG>"}
        {"type": "text",  "text": "..."}

    Messages from Gemini → browser:
        {"type": "audio", "data": "<base64 PCM>"}
        {"type": "transcript", "role": "user"|"assistant", "text": "..."}
        {"type": "tool_call", "name": "...", "args": {...}}
        {"type": "tool_result", "name": "...", "result": {...}}
        {"type": "status", "status": "connected"|"disconnected", "message": "..."}
        {"type": "turn_complete"}
        {"type": "interrupted"}
        {"type": "error", "message": "..."}
    """
    await ws.accept()
    logger.info("WebSocket connection accepted")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        await ws.send_json({
            "type": "error",
            "message": "GEMINI_API_KEY not configured. Set it in backend/.env",
        })
        await ws.close()
        return

    # Callback to send messages back to the browser
    async def send_to_browser(message: str):
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.error("Error sending to browser: %s", e)

    # Create Gemini session
    gemini_session = GeminiLiveSession(
        api_key=api_key,
        send_callback=send_to_browser,
    )

    # Start Gemini session in background
    gemini_task = asyncio.create_task(gemini_session.start())

    try:
        # Wait briefly for session to initialise
        await asyncio.sleep(1)

        # Receive messages from the browser
        while True:
            try:
                raw = await ws.receive_text()
                msg = json.loads(raw)

                msg_type = msg.get("type")

                if msg_type == "audio":
                    audio_bytes = base64.b64decode(msg["data"])
                    await gemini_session.send_audio(audio_bytes)

                elif msg_type == "video":
                    frame_bytes = base64.b64decode(msg["data"])
                    logger.debug("Video frame received: %d bytes", len(frame_bytes))
                    await gemini_session.send_video(frame_bytes)

                elif msg_type == "text":
                    logger.info("Received text from browser: %s", msg["text"][:50])
                    await gemini_session.send_text(msg["text"])

                else:
                    logger.warning("Unknown message type: %s", msg_type)

            except WebSocketDisconnect:
                logger.info("Browser WebSocket disconnected")
                break
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON from browser: %s", e)
            except Exception as e:
                logger.error("Error processing browser message: %s", e)
                break

    finally:
        logger.info("Cleaning up Gemini session")
        await gemini_session.stop()
        gemini_task.cancel()
        try:
            await gemini_task
        except asyncio.CancelledError:
            pass

        try:
            await ws.send_json({
                "type": "status",
                "status": "disconnected",
                "message": "Session ended",
            })
        except Exception:
            pass  # WebSocket may already be closed


# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
