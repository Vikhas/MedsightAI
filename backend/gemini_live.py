"""
MediSight — Gemini Live API Session Wrapper

Manages the bi-directional streaming session with Gemini Live API
using the google-genai SDK. Handles audio/video input, response
receiving, and tool call execution.

Key design: the session auto-reconnects internally when the Gemini
WebSocket dies, keeping the browser WebSocket alive throughout.
"""

import asyncio
import base64
import json
import logging
from pathlib import Path

from google import genai
from google.genai import types

from agent_tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

# Load the system prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text() if PROMPT_PATH.exists() else ""

# Model configuration
MODEL = "gemini-2.5-flash-native-audio-latest"
VOICE = "Aoede"

MAX_RECONNECT_ATTEMPTS = 10


class GeminiLiveSession:
    """Manages a Gemini Live API session with auto-reconnect."""

    def __init__(self, api_key: str, send_callback):
        self.client = genai.Client(api_key=api_key)
        self.send_callback = send_callback
        self.session = None
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._video_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._session_ready = asyncio.Event()
        self._audio_paused = False  # Pause audio during text sends

    async def start(self):
        """Start the session with auto-reconnect loop."""
        self._running = True
        attempt = 0

        while self._running and attempt < MAX_RECONNECT_ATTEMPTS:
            self._session_ready.clear()
            try:
                await self._run_session()
            except asyncio.CancelledError:
                logger.info("Gemini session cancelled")
                break
            except Exception as e:
                logger.error("Gemini session error: %s", e)

            if not self._running:
                break

            # Auto-reconnect
            attempt += 1
            logger.info("Auto-reconnecting (%d/%d)", attempt, MAX_RECONNECT_ATTEMPTS)
            try:
                await self.send_callback(json.dumps({
                    "type": "status",
                    "status": "reconnecting",
                    "message": f"Reconnecting ({attempt}/{MAX_RECONNECT_ATTEMPTS})…",
                }))
            except Exception:
                break

            self._drain_queues()
            await asyncio.sleep(0.5)

        self._running = False
        self.session = None

    async def _run_session(self):
        """Run a single Gemini session until it disconnects."""
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(**decl)
                    for decl in TOOL_DECLARATIONS
                ]
            )
        ]

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE,
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_PROMPT)]
            ),
            tools=tools,
        )

        logger.info("Connecting to Gemini Live API with model=%s", MODEL)

        async with self.client.aio.live.connect(
            model=MODEL,
            config=config,
        ) as session:
            self.session = session
            self._session_ready.set()
            logger.info("Gemini Live session established")

            await self.send_callback(json.dumps({
                "type": "status",
                "status": "connected",
                "message": "MediSight AI connected and ready",
            }))

            # Run concurrent tasks
            self._tasks = [
                asyncio.create_task(self._send_audio_loop()),
                asyncio.create_task(self._send_video_loop()),
                asyncio.create_task(self._receive_loop()),
            ]

            try:
                done, pending = await asyncio.wait(
                    self._tasks,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    exc = task.exception()
                    if exc:
                        raise exc
            finally:
                for task in self._tasks:
                    task.cancel()
                self._tasks.clear()
                self._session_ready.clear()
                self.session = None

    async def stop(self):
        """Stop the session gracefully."""
        self._running = False
        self.session = None
        self._session_ready.clear()
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self._drain_queues()

    def _drain_queues(self):
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._video_queue.empty():
            try:
                self._video_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def send_audio(self, audio_data: bytes):
        if not self._running:
            return
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            pass

    async def send_video(self, frame_data: bytes):
        if not self._running:
            return
        if self._video_queue.full():
            try:
                self._video_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._video_queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            pass

    async def send_text(self, text: str):
        """Send a text message to Gemini.
        
        Pauses audio briefly, sends text with turn_complete, then sends
        a long silence (2s) to trigger the audio model to respond.
        """
        logger.info("send_text called: session=%s, running=%s", 
                     self.session is not None, self._running)
        if not self._running:
            return
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("send_text: timed out waiting for session")
            return

        session = self.session
        if not session:
            return

        try:
            # 1. Pause mic audio
            self._audio_paused = True
            await asyncio.sleep(0.15)
            self._drain_queues()

            # 2. Send the text as a complete user turn  
            logger.info("Sending text to Gemini: %s", text[:80])
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=True,
            )
            logger.info("Text content sent with turn_complete=True")
            
            # 3. Send 2 seconds of silence audio to wake up the native
            #    audio model (16kHz, 16-bit PCM = 32000 bytes per second)
            silence = b'\x00' * 64000  # 2 seconds
            for i in range(0, len(silence), 6400):
                chunk = silence[i:i+6400]
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type="audio/pcm;rate=16000",
                    )
                )
                await asyncio.sleep(0.05)
            logger.info("Silence audio trigger sent (2s)")
            
            # 4. Wait for model to start processing
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.error("Error sending text: %s", e)
        finally:
            # 5. Resume mic audio
            self._audio_paused = False

    # -----------------------------------------------------------------------
    # Internal loops — NO LOCKS, just check session is valid before each send
    # -----------------------------------------------------------------------

    async def _send_audio_loop(self):
        """Send queued audio to Gemini with batching."""
        while self._running:
            try:
                # Skip sending while paused (text turn in progress)
                if self._audio_paused:
                    # Drain queue during pause to prevent buildup
                    while not self._audio_queue.empty():
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    await asyncio.sleep(0.05)
                    continue

                # Wait for audio data
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Batch additional available chunks
                chunks = [chunk]
                while not self._audio_queue.empty() and len(chunks) < 5:
                    try:
                        chunks.append(self._audio_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                session = self.session
                if session and self._running and not self._audio_paused:
                    combined = b''.join(chunks)
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=combined,
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                # Yield to event loop for ping/pong
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Audio send error: %s", e)
                raise

    async def _send_video_loop(self):
        """Send video frames to Gemini."""
        video_count = 0
        while self._running:
            try:
                try:
                    frame_data = await asyncio.wait_for(
                        self._video_queue.get(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    continue

                session = self.session
                if session and self._running:
                    # Save the latest frame for the analyze_camera_frame tool
                    try:
                        with open("/tmp/medisight_last_frame.jpg", "wb") as f:
                            f.write(frame_data)
                    except Exception as e:
                        logger.error("Failed to save frame: %s", e)

                    await session.send_realtime_input(
                        video=types.Blob(
                            data=frame_data,
                            mime_type="image/jpeg",
                        )
                    )
                    video_count += 1
                    if video_count % 30 == 1:
                        logger.info("Video frame #%d sent (%d bytes)", video_count, len(frame_data))
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Video send error: %s", e)
                raise

    async def _receive_loop(self):
        """Receive responses from Gemini and forward to browser.
        
        CRITICAL: session.receive() is a one-turn async generator.
        It stops yielding after the model sends turn_complete.
        We must call it again in a loop for multi-turn conversations.
        """
        try:
            logger.info("Receive loop started")
            while self._running:
                session = self.session
                if not session:
                    await asyncio.sleep(0.1)
                    continue
                try:
                    async for response in session.receive():
                        if not self._running:
                            return
                        await self._handle_response(response)
                    # Generator exhausted (turn_complete) — loop to start next turn
                    logger.info("Turn receive completed, ready for next turn")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Receive error mid-turn: %s", e)
                    raise
            logger.info("Receive loop ended (not running)")
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
        except Exception as e:
            logger.error("Receive loop fatal error: %s", e)
            raise

    async def _handle_response(self, response):
        # Log all response types for debugging
        has_sc = response.server_content is not None
        has_tc = response.tool_call is not None
        logger.debug("Response: server_content=%s, tool_call=%s", has_sc, has_tc)

        if response.server_content:
            sc = response.server_content

            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data:
                        audio_b64 = base64.b64encode(
                            part.inline_data.data
                        ).decode("utf-8")
                        await self.send_callback(json.dumps({
                            "type": "audio",
                            "data": audio_b64,
                        }))

                    if part.text:
                        logger.info("Model text: %s", part.text[:100])
                        await self.send_callback(json.dumps({
                            "type": "transcript",
                            "role": "assistant",
                            "text": part.text,
                        }))

            if hasattr(sc, "input_transcription") and sc.input_transcription:
                if hasattr(sc.input_transcription, "text") and sc.input_transcription.text:
                    logger.info("Input transcription: %s", sc.input_transcription.text[:100])
                    await self.send_callback(json.dumps({
                        "type": "transcript",
                        "role": "user",
                        "text": sc.input_transcription.text,
                    }))

            if hasattr(sc, "output_transcription") and sc.output_transcription:
                if hasattr(sc.output_transcription, "text") and sc.output_transcription.text:
                    logger.info("Output transcription: %s", sc.output_transcription.text[:100])
                    await self.send_callback(json.dumps({
                        "type": "transcript",
                        "role": "assistant",
                        "text": sc.output_transcription.text,
                    }))

            if sc.turn_complete:
                logger.info(">>> TURN COMPLETE <<<")
                await self.send_callback(json.dumps({
                    "type": "turn_complete",
                }))

            if sc.interrupted:
                logger.info(">>> INTERRUPTED <<<")
                await self.send_callback(json.dumps({
                    "type": "interrupted",
                }))

        if response.tool_call:
            await self._handle_tool_calls(response.tool_call)

    async def _handle_tool_calls(self, tool_call):
        function_responses = []

        for fc in tool_call.function_calls:
            logger.info("Tool call: %s(%s)", fc.name, fc.args)

            await self.send_callback(json.dumps({
                "type": "tool_call",
                "name": fc.name,
                "args": fc.args,
            }))

            tool_fn = TOOL_FUNCTIONS.get(fc.name)
            if tool_fn:
                try:
                    result = tool_fn(**fc.args)
                except Exception as e:
                    logger.error("Tool execution error: %s", e)
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {fc.name}"}

            await self.send_callback(json.dumps({
                "type": "tool_result",
                "name": fc.name,
                "result": result,
            }))

            function_responses.append(
                types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response={"result": json.dumps(result)},
                )
            )

        session = self.session
        if session and self._running:
            try:
                await session.send_tool_response(
                    function_responses=function_responses
                )
            except Exception as e:
                logger.error("Error sending tool response: %s", e)
