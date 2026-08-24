"""Step 2: verify LiveInterpreterSession works with real Genesys call audio.

This is the echo test's pipeline, but EchoAudio is replaced by
TranslateAudio: incoming call audio is fed into a real Azure
TranslationRecognizer (via LiveInterpreterSession), and the translated,
synthesized speech is sent back out on the SAME leg. You should hear your
OWN voice translated into the target language, not an echo.

This is still single-leg - it proves the recognizer works against real
telephony audio before we wire two legs together (Step 3).

Endpoint: wss://<your-render-host>/audiohook
Required Render environment variables:
  AZURE_SPEECH_KEY
  AZURE_SPEECH_RESOURCE_NAME   (e.g. voicebots-speech-byos)
Optional:
  AZURE_SPEECH_VOICE           (defaults based on target language)
  TEST_TARGET_LANGUAGE         (defaults to "en")
"""

from __future__ import annotations

import asyncio
import os

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

TARGET_LANGUAGE = os.environ.get("TEST_TARGET_LANGUAGE", "en")


def build_test_config(target_language: str) -> AppConfig:
    """Build an AppConfig straight from Render environment variables -
    same fields sample_code.py's load_config() produces, just sourced
    from os.environ directly instead of argparse/.env (Render sets real
    environment variables, so no dotenv loading is needed here)."""
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
    """Feeds incoming call audio into a LiveInterpreterSession and pushes
    the translated, synthesized audio back out on the same leg."""

    def __init__(self, target_language: str) -> None:
        super().__init__()
        self._target_language = target_language
        self._session: LiveInterpreterSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._out_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._pump_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._loop = asyncio.get_running_loop()
            speechsdk = load_speech_sdk()
            config = build_test_config(self._target_language)
            self._session = LiveInterpreterSession(
                config,
                speechsdk,
                on_translated_audio=self._on_translated_audio,
                label="leg-A",
            )
            self._session.start()
            self._pump_task = self._loop.create_task(self._pump_output())
            return

        if isinstance(frame, InputAudioRawFrame):
            # Feed real Genesys call audio into the recognizer. Do not
            # forward the raw frame downstream - only translated audio
            # (pushed via _pump_output) should reach the output transport.
            if self._session is not None:
                self._session.push_audio(frame.audio)
            return

        if isinstance(frame, EndFrame):
            if self._session is not None:
                self._session.stop()
            if self._pump_task is not None:
                self._pump_task.cancel()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _on_translated_audio(self, audio: bytes) -> None:
        # IMPORTANT: this callback fires from an Azure Speech SDK
        # background thread, not the pipeline's asyncio loop. Never touch
        # asyncio objects directly here - hand off thread-safely instead.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._out_queue.put_nowait, audio)

    async def _pump_output(self) -> None:
        while True:
            audio = await self._out_queue.get()
            await self.push_frame(
                OutputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "endpoint": "wss://<this-host>/audiohook", "target_language": TARGET_LANGUAGE}


@app.websocket("/audiohook/{session_id}")
async def audiohook_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    logger.info(f"Genesys AudioHook connection accepted, session_id={session_id}")

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

    pipeline = Pipeline([transport.input(), TranslateAudio(TARGET_LANGUAGE), transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client) -> None:
        logger.info("AudioHook session closed")
        await task.cancel()

    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
