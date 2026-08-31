"""Step 3: two-leg bridge - cs <-> en live interpretation.

Two independent Genesys AudioHook connections (leg A = Czech speaker,
leg B = English speaker) each run their own TranslateAudio processor.
Instead of playing translated audio back onto the SAME leg (Step 2), each
leg's translated output is routed to its PARTNER leg's output transport,
so each participant hears the other, translated.

Single-pair only (matches the agreed MVP scope): a simple broker pairs
the first connection with the second. If a third connects while a pair
is already active, it becomes the new "waiting" leg for the next pair.

Which language a leg translates INTO is decided by a path segment on the
WSS URL, not by connection order - set this per Architect flow. Genesys
appends "/<call-id>" to whatever URL you configure, so the language must
be a clean path segment BEFORE that, not a query parameter (a query
parameter gets mangled since Genesys just concatenates the call id onto
the end of the string you gave it, "?target=en" + "/<uuid>" -> broken
"?target=en/<uuid>"):

  Flow A (Czech speaker)   -> wss://<host>/audiohook/en
  Flow B (English speaker) -> wss://<host>/audiohook/cs

Required Render environment variables:
  AZURE_SPEECH_KEY
  AZURE_SPEECH_RESOURCE_NAME   (e.g. voicebots-speech-byos)
Optional:
  AZURE_SPEECH_VOICE
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.frames.frames import EndFrame, Frame, InputAudioRawFrame, OutputAudioRawFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.genesys import AudioHookChannel, GenesysAudioHookSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from sample_code import (
    AppConfig,
    build_universal_v2_endpoint,
    load_speech_sdk,
    normalize_target_language,
    resolve_voice_name,
)
from live_interpreter_session import LiveInterpreterSession

app = FastAPI()


def build_leg_config(target_language: str) -> AppConfig:
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    endpoint = build_universal_v2_endpoint(
        resource_name=os.environ.get("AZURE_SPEECH_RESOURCE_NAME"),
        endpoint=os.environ.get("AZURE_SPEECH_ENDPOINT"),
    )
    language = normalize_target_language(target_language)
    voice = resolve_voice_name(language, os.environ.get("AZURE_SPEECH_VOICE"))
    return AppConfig(
        key=key or None,
        endpoint=endpoint,
        target_language=language,
        voice_name=voice,
        input_wav=None,
        output_wav=None,
        timeout_seconds=3600.0,
        play_audio=False,
        auth_mode="key",
        tenant_id=None,
        subscription_id=None,
    )


class TranslateAudio(FrameProcessor):
    """One leg of the bridge: recognizes+translates this leg's audio and
    sends the synthesized result to the PARTNER leg's output, not its own."""

    def __init__(self, target_language: str, label: str) -> None:
        super().__init__()
        self.target_language = target_language
        self.label = label
        self.partner: Optional["TranslateAudio"] = None

        self._session: Optional[LiveInterpreterSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._out_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._task = None  # PipelineTask, injected via set_task() after creation

    def set_task(self, task) -> None:
        self._task = task

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._loop = asyncio.get_running_loop()
            speechsdk = load_speech_sdk()
            config = build_leg_config(self.target_language)
            self._session = LiveInterpreterSession(
                config,
                speechsdk,
                on_translated_audio=self._on_translated_audio,
                label=self.label,
            )
            self._session.start()
            self._pump_task = self._loop.create_task(self._pump_output())
            await broker.register(self)
            return

        if isinstance(frame, InputAudioRawFrame):
            if self._session is not None:
                self._session.push_audio(frame.audio)
            return  # raw audio never goes straight to output on this leg

        if isinstance(frame, EndFrame):
            await self._teardown()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _teardown(self) -> None:
        """Stop this leg's recognizer and unregister from the broker.
        Idempotent - safe to call multiple times from different triggers."""
        if self._shutting_down:
            return
        self._shutting_down = True
        if self._session is not None:
            self._session.stop()
        if self._pump_task is not None:
            self._pump_task.cancel()
        await broker.unregister(self)

    async def handle_disconnect(self) -> None:
        """Called when THIS leg's websocket actually disconnects (Genesys
        closed it, or a network drop) - this is the reliable trigger point,
        not EndFrame, since task.cancel() drives shutdown via CancelFrame
        internally. Cleans up and forces the partner leg to hang up too."""
        partner = self.partner
        await self._teardown()
        if self._task is not None:
            await self._task.cancel()
        if partner is not None:
            logger.info(
                f"[{self.label}] ended - propagating disconnect to partner [{partner.label}]"
            )
            await partner.force_disconnect()

    async def force_disconnect(self) -> None:
        """Called on the PARTNER leg to make ITS OWN Genesys call end too,
        using the same task.cancel() mechanism that reliably closes a leg
        when Genesys ends it directly."""
        await self._teardown()
        if self._task is not None:
            await self._task.cancel()

    def _on_translated_audio(self, audio: bytes) -> None:
        # Fires on an Azure Speech SDK background thread. Route the
        # translated audio to the PARTNER leg's outgoing queue - that is
        # what makes this a cross-wired bridge instead of an echo.
        if self.partner is None:
            logger.warning(f"[{self.label}] no partner yet, dropping translated audio")
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.partner._out_queue.put_nowait, audio)

    async def _pump_output(self) -> None:
        while True:
            audio = await self._out_queue.get()
            await self.push_frame(
                OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )


class PairBroker:
    """Pairs the first waiting leg with the next one that connects.
    Single pair at a time - matches the agreed proof-of-concept scope."""

    STALE_AFTER_SECONDS = 120.0

    def __init__(self) -> None:
        self._waiting: Optional[TranslateAudio] = None
        self._waiting_since: float = 0.0
        self._lock = asyncio.Lock()

    async def register(self, leg: TranslateAudio) -> None:
        async with self._lock:
            if self._waiting is not None:
                age = asyncio.get_running_loop().time() - self._waiting_since
                if age > self.STALE_AFTER_SECONDS:
                    logger.warning(
                        f"[{self._waiting.label}] was waiting {age:.0f}s with no partner - "
                        f"treating as stale, discarding before pairing [{leg.label}]"
                    )
                    self._waiting = None

            if self._waiting is None:
                self._waiting = leg
                self._waiting_since = asyncio.get_running_loop().time()
                logger.info(f"[{leg.label}] connected, waiting for partner leg...")
            else:
                partner = self._waiting
                self._waiting = None
                leg.partner = partner
                partner.partner = leg
                logger.info(f"[{leg.label}] paired with [{partner.label}]")

    async def unregister(self, leg: TranslateAudio) -> None:
        async with self._lock:
            if self._waiting is leg:
                self._waiting = None
            if leg.partner is not None:
                leg.partner.partner = None
                logger.info(f"[{leg.label}] disconnected, unpairing from [{leg.partner.label}]")
                leg.partner = None


broker = PairBroker()


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "endpoint": "wss://<this-host>/audiohook/en or /audiohook/cs"}


@app.websocket("/audiohook/{target_language}/{session_id}")
async def audiohook_endpoint(websocket: WebSocket, target_language: str, session_id: str) -> None:
    label = f"leg[target={target_language}]-{session_id[:8]}"

    await websocket.accept()
    logger.info(f"[{label}] Genesys AudioHook connection accepted")

    serializer = GenesysAudioHookSerializer(
        params=GenesysAudioHookSerializer.InputParams(
            channel=AudioHookChannel.EXTERNAL,
        )
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=serializer,
            audio_out_fixed_packet_size=1600,
        ),
    )

    processor = TranslateAudio(target_language, label)
    pipeline = Pipeline([transport.input(), processor, transport.output()])
    task = PipelineTask(pipeline)
    processor.set_task(task)
    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client) -> None:
        logger.info(f"[{label}] AudioHook session closed")
        # This is the reliable trigger point (see handle_disconnect
        # docstring for why EndFrame is not) - cleans up this leg AND
        # forces the partner leg's call to end too.
        await processor.handle_disconnect()

    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
