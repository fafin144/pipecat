"""Step 1: verify the Genesys AudioHook <-> Pipecat transport works.

This deliberately does NOT include translation yet. It is the simplest
possible pipeline: audio arriving from Genesys on this WebSocket is
immediately sent back out on the same connection. If this works end to
end (Call Audio Connector in Architect connects, handshake succeeds,
and you hear your own voice echoed back on the test leg), we know the
protocol/auth/transport layer is solid, and translation can be wired in
next using LiveInterpreterSession without protocol-layer surprises.

Endpoint: wss://<your-render-host>/audiohook
"""

from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.genesys import AudioHookChannel, GenesysAudioHookSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

app = FastAPI()


class EchoAudio(FrameProcessor):
    """Turns incoming InputAudioRawFrame into OutputAudioRawFrame so the
    output transport is actually allowed to send it back out. Everything
    else (StartFrame, EndFrame, etc.) is passed through unchanged."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                ),
                direction,
            )
        else:
            await self.push_frame(frame, direction)


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "endpoint": "wss://<this-host>/audiohook"}


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
            # Required by Genesys: prevents 429 rate limiting on outbound audio.
            audio_out_fixed_packet_size=1600,
        ),
    )

    # Simplest possible pipeline: whatever comes in on this leg goes
    # straight back out on the same leg. Pure echo, no translation yet.
    # EchoAudio converts InputAudioRawFrame -> OutputAudioRawFrame, which
    # is required - output transports only ever send OutputAudioRawFrame.
    pipeline = Pipeline([transport.input(), EchoAudio(), transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def on_connected(_transport, _client) -> None:
        logger.info(
            "AudioHook session open. participant=%s input_variables=%s",
            getattr(serializer, "participant", None),
            getattr(serializer, "input_variables", None),
        )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client) -> None:
        logger.info("AudioHook session closed")
        await task.cancel()

    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
