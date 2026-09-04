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
  ANTHROPIC_API_KEY            (for the "Hey AI Agent" assistant)
Optional:
  AZURE_SPEECH_VOICE
  AGENT_VOICE                  (voice for the AI agent's spoken answers,
                                 default: en-US-JennyNeural)

Two trigger phrases, either can be said on either leg:
  "Hey Translate" -> activates bidirectional live translation (both legs,
                      persists for the rest of the call)
  "Hey AI Agent"   -> asks a question (either in the same utterance, e.g.
                      "Hey AI Agent, what's the weather in London", or in
                      the next utterance if said alone) and plays Claude's
                      answer to BOTH legs, muting normal audio meanwhile
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import FastAPI, WebSocket
from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    StartFrame,
)
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
from agent_assistant import ask_agent

app = FastAPI()

TRIGGER_PHRASE = "hey translate"
AGENT_TRIGGER_PHRASE = "hey ai agent"


def _normalize(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return " ".join(normalized.split())


def _contains_trigger(text: str, phrase: str) -> bool:
    """Case/punctuation-insensitive substring check for a trigger phrase."""
    return phrase in _normalize(text)


def _extract_after_trigger(text: str, phrase: str) -> Optional[str]:
    """If `phrase` appears in `text`, return whatever comes after it
    (may be an empty string if the trigger was the entire utterance).
    Returns None if the phrase isn't present at all."""
    normalized = _normalize(text)
    idx = normalized.find(phrase)
    if idx == -1:
        return None
    return normalized[idx + len(phrase):].strip()


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


async def synthesize_speech(text: str) -> bytes:
    """Plain text-to-speech for the agent's answer (separate from the
    TranslationRecognizer sessions, which are for cs<->en interpretation).

    Uses the resource's standard endpoint (not the /stt/speech/universal/v2
    path used for Live Interpreter) - a plain SpeechSynthesizer is not
    guaranteed to work against that specialized path. Runs the blocking
    Speech SDK call in a thread so it doesn't block the event loop.
    """
    speechsdk = load_speech_sdk()
    key = os.environ.get("AZURE_SPEECH_KEY", "").strip()
    resource_name = os.environ.get("AZURE_SPEECH_RESOURCE_NAME", "").strip()
    voice = os.environ.get("AGENT_VOICE", "en-US-JennyNeural")

    def _synth() -> bytes:
        speech_config = speechsdk.SpeechConfig(
            subscription=key,
            endpoint=f"https://{resource_name}.cognitiveservices.azure.com/",
        )
        speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)
        return b""

    return await asyncio.to_thread(_synth)


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
        self._serializer = None  # GenesysAudioHookSerializer, injected via set_serializer()
        self.translation_active = False
        self.agent_busy = False
        self.awaiting_agent_question = False

    def set_task(self, task) -> None:
        self._task = task

    def set_serializer(self, serializer) -> None:
        self._serializer = serializer

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
                on_recognizing=self._check_translate_trigger,
                on_recognized=self._on_recognized_final,
                label=self.label,
            )
            self._session.start()
            self._pump_task = self._loop.create_task(self._pump_output())
            await broker.register(self)
            return

        if isinstance(frame, InputAudioRawFrame):
            # Always feed the recognizer - it's what watches for the
            # trigger phrase, and once active, what produces translated
            # audio. It runs continuously across both phases; nothing
            # about it needs to restart when the mode switches.
            if self._session is not None:
                self._session.push_audio(frame.audio)

            if not self.translation_active and not self.agent_busy and self.partner is not None:
                # Phase 1: no translation yet - relay this leg's raw audio
                # straight through to the partner, untranslated, so the
                # call sounds completely normal until the trigger fires.
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self.partner._out_queue.put_nowait, frame.audio
                    )
            return

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
        try:
            partner = self.partner
            await self._teardown()
            if self._task is not None:
                await self._task.cancel()
            if partner is not None:
                logger.info(
                    f"[{self.label}] ended - propagating disconnect to partner [{partner.label}]"
                )
                await partner.force_disconnect()
            else:
                logger.info(f"[{self.label}] ended - no partner was set, nothing to propagate")
        except Exception:
            logger.exception(f"[{self.label}] handle_disconnect failed")

    async def force_disconnect(self) -> None:
        """Called on the PARTNER leg to make ITS OWN Genesys call end too.

        The automatic EndFrame/CancelFrame -> disconnect-message handling
        in GenesysAudioHookSerializer defaults to action="transfer", which
        tells Genesys to expect the call to be handed off elsewhere -
        Genesys then waits (~20s observed) before giving up and hanging up
        on its own, instead of ending the call immediately. Sending our
        own disconnect message with action="finished" first avoids that
        wait entirely.
        """
        try:
            logger.info(f"[{self.label}] force_disconnect invoked by partner")
            if self._serializer is not None:
                disconnect_msg = self._serializer.create_disconnect_message(
                    reason="completed", action="finished"
                )
                await self.push_frame(
                    OutputTransportMessageFrame(message=disconnect_msg),
                    FrameDirection.DOWNSTREAM,
                )
                # Give the message a moment to actually go out over the
                # wire before we tear down the transport underneath it.
                await asyncio.sleep(0.1)
            await self._teardown()
            if self._task is not None:
                await self._task.cancel()
            logger.info(f"[{self.label}] force_disconnect completed")
        except Exception:
            logger.exception(f"[{self.label}] force_disconnect failed")

    def _on_translated_audio(self, audio: bytes) -> None:
        # Fires on an Azure Speech SDK background thread. Route the
        # translated audio to the PARTNER leg's outgoing queue - that is
        # what makes this a cross-wired bridge instead of an echo.
        if not self.translation_active or self.agent_busy:
            return  # Phase 1, or the AI agent is currently speaking - either
            # way, this leg's synthesized output should not go out right now.
        if self.partner is None:
            logger.warning(f"[{self.label}] no partner yet, dropping translated audio")
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.partner._out_queue.put_nowait, audio)

    def _check_translate_trigger(self, text: str) -> None:
        # Called from LiveInterpreterSession's recognizing callback (partial
        # results) - fires on an Azure Speech SDK background thread, so hop
        # back onto the pipeline's event loop via run_coroutine_threadsafe.
        if self.translation_active:
            return
        if _contains_trigger(text, TRIGGER_PHRASE) and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._activate_translation(), self._loop)

    def _on_recognized_final(self, text: str) -> None:
        # Final recognized text: re-check the translate trigger (in case it
        # was missed on partial results), and separately handle the AI
        # agent flow, which needs a clean, complete utterance.
        self._check_translate_trigger(text)

        if self._loop is None:
            return

        if self.awaiting_agent_question:
            asyncio.run_coroutine_threadsafe(self._handle_agent_question(text), self._loop)
            return

        remainder = _extract_after_trigger(text, AGENT_TRIGGER_PHRASE)
        if remainder is None:
            return  # agent trigger phrase not present in this utterance
        if remainder:
            # "Hey AI Agent, what's the weather in London" said in one go.
            asyncio.run_coroutine_threadsafe(self._handle_agent_question(remainder), self._loop)
        else:
            # Trigger said alone - capture the NEXT utterance as the question.
            self.awaiting_agent_question = True
            logger.info(f"[{self.label}] AI agent trigger detected, awaiting question...")

    async def _handle_agent_question(self, question: str) -> None:
        self.awaiting_agent_question = False
        logger.info(f"[{self.label}] asking AI agent: {question!r}")
        self.agent_busy = True
        if self.partner is not None:
            self.partner.agent_busy = True
        try:
            answer = await ask_agent(question)
        except Exception:
            logger.exception(f"[{self.label}] agent call failed")
            answer = "Sorry, something went wrong answering that."
        logger.info(f"[{self.label}] agent answered: {answer!r}")
        try:
            audio = await synthesize_speech(answer)
            if audio:
                # Both parties hear the answer - push to this leg's own
                # output queue AND the partner's, since we're already
                # running on the pipeline's event loop here (no thread
                # hop needed, unlike the SDK-thread callbacks above).
                self._out_queue.put_nowait(audio)
                if self.partner is not None:
                    self.partner._out_queue.put_nowait(audio)
            else:
                logger.warning(f"[{self.label}] agent TTS produced no audio")
        except Exception:
            logger.exception(f"[{self.label}] agent TTS failed")
        finally:
            self.agent_busy = False
            if self.partner is not None:
                self.partner.agent_busy = False

    async def _activate_translation(self) -> None:
        """Trigger phrase detected on THIS leg - activate translation on
        both legs simultaneously so mode switches for the whole call."""
        if self.translation_active:
            return
        self.translation_active = True
        logger.info(f"[{self.label}] trigger phrase detected - translation now ACTIVE")
        if self.partner is not None:
            await self.partner.activate_from_partner()

    async def activate_from_partner(self) -> None:
        """Called on THIS leg when the PARTNER leg detected the trigger,
        so both legs switch from raw passthrough to translation together."""
        if self.translation_active:
            return
        self.translation_active = True
        logger.info(f"[{self.label}] translation now ACTIVE (triggered by partner)")

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
    processor.set_serializer(serializer)
    pipeline = Pipeline([transport.input(), processor, transport.output()])
    task = PipelineTask(pipeline)
    processor.set_task(task)
    runner = PipelineRunner()

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client) -> None:
        logger.info(f"[{label}] AudioHook session closed")
        try:
            # This is the reliable trigger point (see handle_disconnect
            # docstring for why EndFrame is not) - cleans up this leg AND
            # forces the partner leg's call to end too.
            await processor.handle_disconnect()
        except Exception:
            logger.exception(f"[{label}] on_disconnected handler failed")

    await runner.run(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
