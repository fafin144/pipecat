"""Live Interpreter session engine, decoupled from local mic/speaker hardware.

This is a refactor of the working `sample_code.py` session logic. The Azure
TranslationRecognizer configuration (create_translation_components) and the
event handling (recognizing/recognized/synthesizing/canceled) are unchanged.
What changes is the I/O boundary:

  - Input:  instead of AudioConfig(use_default_microphone=True), audio is
            pushed in programmatically via push_audio(bytes) into a
            PushAudioInputStream (expects 16 kHz, 16-bit, mono PCM - the
            Speech SDK's default streaming format).
  - Output: instead of LiveAudioPlayer writing to a sound device, the raw
            synthesized PCM bytes from the `synthesizing` event are handed
            to an on_translated_audio(bytes) callback you provide. Those
            bytes are also 16 kHz / 16-bit / mono PCM (same output format
            sample_code.py configures via Raw16Khz16BitMonoPcm).

This class does not know anything about Genesys or WebSockets - it is the
translation engine only. One instance = one direction of translation for
one participant leg (it auto-detects that leg's spoken language and
synthesizes into config.target_language). For a two-way bridge you run two
instances, one per leg, and cross-wire each one's on_translated_audio into
the *other* leg's outbound audio.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Reuse the exact, already-working configuration/session-building logic.
from sample_code import (
    AppConfig,
    ConfigurationError,
    SessionError,
    create_translation_components,
    format_canceled_event,
    format_recognized_event,
    format_recognizing_event,
)


class LiveInterpreterSession:
    """One TranslationRecognizer instance fed by a push stream instead of a mic."""

    def __init__(
        self,
        config: AppConfig,
        speechsdk: Any,
        on_translated_audio: Callable[[bytes], None],
        on_recognizing: Optional[Callable[[str], None]] = None,
        on_recognized: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        label: str = "session",
    ) -> None:
        self._speechsdk = speechsdk
        self._on_translated_audio = on_translated_audio
        self._on_recognizing = on_recognizing
        self._on_recognized = on_recognized
        self._on_error = on_error
        self.label = label

        translation_config, auto_detect_config = create_translation_components(
            config, speechsdk
        )

        # Default PushAudioInputStream format is 16kHz/16-bit/mono PCM -
        # matches what push_audio() expects callers to provide.
        self._push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        self._recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config,
        )

        self._recognizer.recognizing.connect(self._handle_recognizing)
        self._recognizer.recognized.connect(self._handle_recognized)
        self._recognizer.synthesizing.connect(self._handle_synthesizing)
        self._recognizer.canceled.connect(self._handle_canceled)
        self._recognizer.session_started.connect(
            lambda _evt: print(f"[{self.label}] SESSION: started")
        )
        self._recognizer.session_stopped.connect(
            lambda _evt: print(f"[{self.label}] SESSION: stopped")
        )

        self._started = False

    # --- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._recognizer.start_continuous_recognition()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._recognizer.stop_continuous_recognition()
        finally:
            self._push_stream.close()
            self._started = False

    # --- input -----------------------------------------------------------

    def push_audio(self, pcm16_mono_16khz: bytes) -> None:
        """Feed raw 16kHz/16-bit/mono PCM audio into the recognizer.

        Call this for every chunk of decoded audio arriving from Genesys
        for this leg (after AudioHook's PCMU/L16 8kHz audio has been
        resampled up to 16kHz mono PCM - see the transport layer).
        """
        if not self._started:
            raise RuntimeError(f"[{self.label}] session not started - call start() first")
        if pcm16_mono_16khz:
            self._push_stream.write(pcm16_mono_16khz)

    # --- event handlers --------------------------------------------------

    def _handle_recognizing(self, event: Any) -> None:
        line = format_recognizing_event(event)
        print(f"[{self.label}] {line}")
        if self._on_recognizing:
            self._on_recognizing(line)

    def _handle_recognized(self, event: Any) -> None:
        line = format_recognized_event(event, self._speechsdk)
        print(f"[{self.label}] {line}")
        if self._on_recognized:
            self._on_recognized(line)

    def _handle_synthesizing(self, event: Any) -> None:
        audio = bytes(getattr(event.result, "audio", b"") or b"")
        if audio:
            self._on_translated_audio(audio)

    def _handle_canceled(self, event: Any) -> None:
        message = format_canceled_event(event)
        print(f"[{self.label}] {message}")
        if self._on_error:
            self._on_error(message)
