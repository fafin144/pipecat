"""Small Azure Speech Live Interpreter demo.

The Speech SDK is imported lazily so configuration and help remain usable on
machines that do not have the native Speech SDK or an audio device installed.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit, urlunsplit


PERSONAL_VOICE = "personal-voice"
UNIVERSAL_V2_PATH = "/stt/speech/universal/v2"
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH = 2
OUTPUT_SAMPLE_RATE = 16000
DEFAULT_TARGET_LANGUAGE = "fr"
DEFAULT_PREBUILT_VOICES = {
    "fr": "fr-FR-DeniseNeural",
    "en": "en-US-JennyNeural",
    "cs": "cs-CZ-AntoninNeural",
    "cz": "cs-CZ-AntoninNeural",
}
SYNTHESIS_LOCALES = {
    "fr": "fr-FR",
    "en": "en-US",
    "cs": "cs-CZ",
}
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_ENV_FILE = Path(".env")
AUTH_MODES = ("auto", "azure-cli", "key")


class ConfigurationError(ValueError):
    """Raised when local configuration cannot start a session."""


class SessionError(RuntimeError):
    """Raised when the Speech SDK reports a session failure."""


@dataclass(frozen=True)
class AppConfig:
    key: str | None
    endpoint: str
    target_language: str
    voice_name: str
    input_wav: Path | None
    output_wav: Path | None
    timeout_seconds: float
    play_audio: bool
    auth_mode: str = "key"
    tenant_id: str | None = None
    subscription_id: str | None = None


def build_universal_v2_endpoint(
    resource_name: str | None = None, endpoint: str | None = None
) -> str:
    """Return the resource-specific universal v2 WebSocket endpoint."""

    if endpoint:
        try:
            parsed = urlsplit(endpoint.strip())
            hostname = parsed.hostname
        except ValueError as exc:
            raise ConfigurationError(
                "AZURE_SPEECH_ENDPOINT must be a valid https:// or wss:// endpoint."
            ) from exc
        if parsed.scheme not in {"wss", "https"} or not parsed.netloc:
            raise ConfigurationError(
                "AZURE_SPEECH_ENDPOINT must be an https:// or wss:// endpoint."
            )
        if (
            not hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError(
                "AZURE_SPEECH_ENDPOINT must not contain credentials, query parameters, or fragments."
            )
        path = parsed.path.rstrip("/")
        if path and path != UNIVERSAL_V2_PATH:
            raise ConfigurationError(
                f"AZURE_SPEECH_ENDPOINT must use {UNIVERSAL_V2_PATH}."
            )
        return urlunsplit(
            (
                "wss",
                parsed.netloc,
                UNIVERSAL_V2_PATH,
                "",
                "",
            )
        )

    name = (resource_name or "").strip()
    if not name:
        raise ConfigurationError(
            "Set AZURE_SPEECH_RESOURCE_NAME or AZURE_SPEECH_ENDPOINT."
        )
    if not all(character.isalnum() or character == "-" for character in name):
        raise ConfigurationError(
            "AZURE_SPEECH_RESOURCE_NAME may contain only letters, numbers, and hyphens."
        )
    return f"wss://{name}.cognitiveservices.azure.com{UNIVERSAL_V2_PATH}"


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Timeout must be a positive number of seconds.") from exc
    if timeout <= 0 or not math.isfinite(timeout):
        raise ConfigurationError("Timeout must be a positive number of seconds.")
    return timeout


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = os.fspath(value).strip()
    return Path(text) if text else None


def _optional_path_from_env(name: str) -> Path | None:
    return _optional_path(os.getenv(name))


def resolve_voice_name(target_language: str, requested_voice: str | None = None) -> str:
    """Return an explicit voice or the prebuilt default for the target language."""

    voice_name = (requested_voice or "").strip()
    if voice_name:
        return voice_name
    language = target_language.strip().lower().split("-", 1)[0]
    try:
        return DEFAULT_PREBUILT_VOICES[language]
    except KeyError as exc:
        supported = ", ".join(sorted({"fr", "en", "cs"}))
        raise ConfigurationError(
            f"No default prebuilt voice for target language {target_language!r}. "
            f"Set --voice or AZURE_SPEECH_VOICE. Default languages: {supported}."
        ) from exc


def normalize_target_language(target_language: str) -> str:
    """Normalize common aliases to Azure speech-translation language codes."""

    language = target_language.strip()
    return "cs" if language.lower() == "cz" else language


def load_env_file(path: Path, required: bool = False) -> bool:
    """Load a dotenv file without overriding variables already in the process."""

    if not path.exists():
        if required:
            raise ConfigurationError(f"Environment file does not exist: {path}")
        return False
    if not path.is_file():
        raise ConfigurationError(f"Environment file is not a file: {path}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ConfigurationError(
            "Environment-file support requires python-dotenv. "
            "Run: python3 -m pip install -r requirements.txt"
        ) from exc
    load_dotenv(dotenv_path=path, override=False)
    return True


def _load_requested_environment(argv: list[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--env-file", type=Path)
    group.add_argument("--no-env-file", action="store_true")
    known, _ = parser.parse_known_args(argv)

    if known.no_env_file:
        return None
    configured_path = os.getenv("AZURE_SPEECH_ENV_FILE", "").strip()
    path = known.env_file or (Path(configured_path) if configured_path else DEFAULT_ENV_FILE)
    required = known.env_file is not None or bool(configured_path)
    if "-h" not in argv and "--help" not in argv:
        load_env_file(path, required=required)
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    env_file = _load_requested_environment(raw_argv)
    parser = argparse.ArgumentParser(
        description="Translate microphone or WAV speech with Azure Speech Live Interpreter."
    )
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--env-file",
        type=Path,
        default=env_file,
        help=(
            "Load configuration from this dotenv file before reading environment "
            "defaults (default: .env or AZURE_SPEECH_ENV_FILE)."
        ),
    )
    env_group.add_argument(
        "--no-env-file",
        action="store_true",
        help="Do not load a dotenv configuration file.",
    )
    parser.add_argument(
        "--resource-name",
        default=os.getenv("AZURE_SPEECH_RESOURCE_NAME"),
        help="Speech resource name (or AZURE_SPEECH_RESOURCE_NAME).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_SPEECH_ENDPOINT"),
        help="Resource endpoint (or AZURE_SPEECH_ENDPOINT).",
    )
    parser.add_argument(
        "--auth-mode",
        choices=AUTH_MODES,
        default=os.getenv("AZURE_SPEECH_AUTH_MODE", "auto").strip().lower(),
        help=(
            "Authentication mode: auto uses a non-empty AZURE_SPEECH_KEY when "
            "available, otherwise Azure CLI; key or azure-cli selects one explicitly."
        ),
    )
    parser.add_argument(
        "--target-language",
        default=os.getenv("AZURE_SPEECH_TARGET_LANGUAGE", DEFAULT_TARGET_LANGUAGE),
        help="Target language code, such as fr, en, or cs (or AZURE_SPEECH_TARGET_LANGUAGE).",
    )
    parser.add_argument(
        "--voice",
        default=os.getenv("AZURE_SPEECH_VOICE"),
        help=(
            "Speech synthesis voice (or AZURE_SPEECH_VOICE). Defaults to a prebuilt "
            "voice for fr, en, or cs. Use 'personal-voice' explicitly for Personal Voice."
        ),
    )
    parser.add_argument(
        "--wav",
        dest="input_wav",
        type=_optional_path,
        default=_optional_path_from_env("AZURE_SPEECH_INPUT_WAV"),
        help="Read speech from a WAV file instead of the default microphone.",
    )
    parser.add_argument(
        "--output-wav",
        type=_optional_path,
        default=_optional_path_from_env("AZURE_SPEECH_OUTPUT_WAV"),
        help="Write synthesized translated audio to this WAV file.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=_positive_timeout(
            os.getenv("AZURE_SPEECH_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
        ),
        help="Maximum session duration in seconds (or AZURE_SPEECH_TIMEOUT).",
    )
    parser.add_argument(
        "--no-play-audio",
        action="store_true",
        help="Do not attempt local playback of synthesized audio.",
    )
    return parser.parse_args(raw_argv)


def load_config(args: argparse.Namespace, environ: dict[str, str] | None = None) -> AppConfig:
    env = os.environ if environ is None else environ
    key = env.get("AZURE_SPEECH_KEY", "").strip()
    requested_auth_mode = str(getattr(args, "auth_mode", "auto") or "auto").strip().lower()
    if requested_auth_mode not in AUTH_MODES:
        valid_modes = ", ".join(AUTH_MODES)
        raise ConfigurationError(
            f"Invalid authentication mode {requested_auth_mode!r}; choose {valid_modes}."
        )
    if requested_auth_mode == "auto":
        auth_mode = "key" if key else "azure-cli"
    else:
        auth_mode = requested_auth_mode
    if auth_mode == "key" and not key:
        raise ConfigurationError(
            "Authentication mode 'key' requires a non-empty AZURE_SPEECH_KEY."
        )
    if auth_mode == "azure-cli" and not env.get("AZURE_CONFIG_DIR", "").strip():
        raise ConfigurationError(
            'Azure CLI authentication requires AZURE_CONFIG_DIR. '
            'Set it to the target Azure CLI profile directory and verify '
            "that Azure CLI is logged in."
        )
    target_language = normalize_target_language(str(args.target_language or ""))
    if not target_language:
        raise ConfigurationError("A target language is required.")
    voice_name = resolve_voice_name(target_language, getattr(args, "voice", None))
    timeout_seconds = _positive_timeout(args.timeout)
    input_wav = _optional_path(args.input_wav)
    if input_wav is not None:
        if not input_wav.is_file():
            raise ConfigurationError(f"WAV input file does not exist: {input_wav}")
        if input_wav.suffix.lower() != ".wav":
            raise ConfigurationError("WAV input must have a .wav extension.")
    return AppConfig(
        key=key or None,
        endpoint=build_universal_v2_endpoint(args.resource_name, args.endpoint),
        target_language=target_language,
        voice_name=voice_name,
        input_wav=input_wav,
        output_wav=_optional_path(args.output_wav),
        timeout_seconds=timeout_seconds,
        play_audio=not args.no_play_audio,
        auth_mode=auth_mode,
        tenant_id=env.get("AZURE_TENANT_ID", "").strip() or None,
        subscription_id=env.get("AZURE_SUBSCRIPTION_ID", "").strip() or None,
    )


def load_speech_sdk() -> Any:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise ConfigurationError(
            "The Speech SDK is not installed. Run: python3 -m pip install -r requirements.txt"
        ) from exc
    return speechsdk


def create_azure_cli_credential(
    tenant_id: str | None = None,
    subscription_id: str | None = None,
) -> Any:
    """Construct the Azure CLI credential only when Entra auth is selected."""

    if not os.getenv("AZURE_CONFIG_DIR", "").strip():
        raise ConfigurationError(
            'Azure CLI authentication requires AZURE_CONFIG_DIR. '
            'Set it to the target Azure CLI profile directory and verify '
            "that Azure CLI is logged in."
        )
    try:
        from azure.identity import AzureCliCredential
    except ImportError as exc:
        raise ConfigurationError(
            "Azure CLI authentication requires azure-identity. "
            "Run: python3 -m pip install -r requirements.txt"
        ) from exc
    credential_options = {}
    if subscription_id:
        credential_options["subscription"] = subscription_id
    elif tenant_id:
        credential_options["tenant_id"] = tenant_id
    return AzureCliCredential(**credential_options)


def create_translation_components(
    config: AppConfig, speechsdk: Any
) -> tuple[Any, Any]:
    """Create the SDK config and open-range language detector."""

    auth_mode = config.auth_mode.strip().lower()
    if auth_mode == "auto":
        auth_mode = "key" if config.key else "azure-cli"
    if auth_mode == "key":
        if not config.key:
            raise ConfigurationError(
                "Authentication mode 'key' requires a non-empty AZURE_SPEECH_KEY."
            )
        sdk_kwargs = {"subscription": config.key, "endpoint": config.endpoint}
    elif auth_mode == "azure-cli":
        sdk_kwargs = {
            "token_credential": create_azure_cli_credential(
                config.tenant_id,
                config.subscription_id,
            ),
            "endpoint": config.endpoint,
        }
    else:
        valid_modes = ", ".join(AUTH_MODES)
        raise ConfigurationError(f"Invalid authentication mode {auth_mode!r}; choose {valid_modes}.")
    translation_config = speechsdk.translation.SpeechTranslationConfig(**sdk_kwargs)
    translation_config.add_target_language(config.target_language)
    translation_config.voice_name = config.voice_name
    language = config.target_language.lower().split("-", 1)[0]
    synthesis_locale = SYNTHESIS_LOCALES.get(language)
    if synthesis_locale is not None:
        translation_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_SynthLanguage,
            synthesis_locale,
        )
    output_format = getattr(
        getattr(speechsdk, "SpeechSynthesisOutputFormat", None),
        "Raw16Khz16BitMonoPcm",
        None,
    )
    if output_format is None:
        raise ConfigurationError(
            "The installed Speech SDK does not support raw 16-kHz PCM synthesis; "
            "upgrade azure-cognitiveservices-speech."
        )
    translation_config.set_speech_synthesis_output_format(output_format)

    # Restrict auto-detection to the languages we actually expect on a call
    # leg (Czech/English), instead of open-range detection across every
    # language Azure supports. This improves both accuracy and latency
    # when we already know the candidate set.
    auto_detect_type = speechsdk.languageconfig.AutoDetectSourceLanguageConfig
    try:
        auto_detect_config = auto_detect_type(languages=["cs-CZ", "en-US"])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "The installed Speech SDK does not support constrained language "
            "auto-detection; upgrade azure-cognitiveservices-speech."
        ) from exc
    return translation_config, auto_detect_config


def format_canceled_event(event: Any) -> str:
    details = getattr(event, "cancellation_details", None)
    if details is None:
        return "CANCELED: Speech service canceled the session."
    reason = getattr(details, "reason", "unknown")
    error_details = getattr(details, "error_details", "")
    message = f"CANCELED: reason={reason}"
    if error_details:
        message += f"; details={error_details}"
    detail_text = f"{reason} {error_details}".lower()
    if "live interpreter" in detail_text or "personal voice" in detail_text:
        message += (
            " Verify that Live Interpreter access and Personal Voice approval are "
            "enabled on this exact Speech resource."
        )
    elif (
        any(code in detail_text for code in ("401", "403"))
        or "unauthorized" in detail_text
        or "not authorized" in detail_text
        or "forbidden" in detail_text
        or "authorization" in detail_text
        or "rbac" in detail_text
        or "access denied" in detail_text
        or "permission denied" in detail_text
    ):
        message += (
            " Verify the Azure identity and Cognitive Services Speech User role "
            "for this Speech resource."
        )
    return message


def _translations(result: Any) -> list[str]:
    translations = getattr(result, "translations", {}) or {}
    return [f"{language}: {text}" for language, text in translations.items() if text]


def format_recognizing_event(event: Any) -> str:
    result = event.result
    translated = "; ".join(_translations(result))
    suffix = f" -> {translated}" if translated else ""
    return f"RECOGNIZING: {getattr(result, 'text', '').strip()}{suffix}"


def format_recognized_event(event: Any, speechsdk: Any) -> str:
    result = event.result
    reason = getattr(result, "reason", None)
    translated = "; ".join(_translations(result))
    if reason == getattr(speechsdk.ResultReason, "TranslatedSpeech", object()):
        return f"RECOGNIZED: {getattr(result, 'text', '').strip()} -> {translated}"
    if reason == getattr(speechsdk.ResultReason, "NoMatch", object()):
        return "NOMATCH: no speech could be recognized."
    return f"RECOGNIZED: {getattr(result, 'text', '').strip()}"


class AudioCollector:
    """Collect synthesizing event bytes and save a valid WAV at session end."""

    def __init__(self, output_wav: Path | None) -> None:
        self.output_wav = output_wav
        self._chunks: list[bytes] = []

    def add(self, audio: bytes) -> None:
        if audio:
            self._chunks.append(bytes(audio))

    def finish(self) -> None:
        if not self._chunks:
            return
        wav_bytes = _as_wav(self._chunks)
        if self.output_wav is not None:
            self.output_wav.parent.mkdir(parents=True, exist_ok=True)
            self.output_wav.write_bytes(wav_bytes)
            print(f"AUDIO: wrote {self.output_wav}")


class LiveAudioPlayer:
    """Play synthesized PCM chunks in order without blocking SDK callbacks."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._error: Exception | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stream: Any | None = None
        self._sounddevice: Any | None = None
        if not enabled:
            return
        try:
            import sounddevice
        except ImportError as exc:
            raise ConfigurationError(
                "Live audio playback requires sounddevice. "
                "Run: python3 -m pip install -r requirements.txt"
            ) from exc
        self._sounddevice = sounddevice
        try:
            self._stream = sounddevice.RawOutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=OUTPUT_CHANNELS,
                dtype="int16",
            )
            self._stream.start()
        except (OSError, RuntimeError, ValueError, sounddevice.PortAudioError) as exc:
            raise ConfigurationError(
                f"Unable to open the default audio output device: {exc}"
            ) from exc
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, audio: bytes) -> None:
        if self.enabled and audio:
            self._queue.put(bytes(audio))

    def finish(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join()
        try:
            self._stream.stop()
            self._stream.close()
        except (
            AttributeError,
            OSError,
            RuntimeError,
            ValueError,
            self._sounddevice.PortAudioError,
        ) as exc:
            if self._error is None:
                self._error = exc
        if self._error is not None:
            raise SessionError(f"Live audio playback failed: {self._error}")

    def _run(self) -> None:
        try:
            while (audio := self._queue.get()) is not None:
                self._stream.write(audio)
        except (
            AttributeError,
            OSError,
            RuntimeError,
            ValueError,
            self._sounddevice.PortAudioError,
        ) as exc:
            self._error = exc


def _as_wav(chunks: list[bytes]) -> bytes:
    if chunks and all(chunk.startswith(b"RIFF") for chunk in chunks):
        try:
            import io

            first_params: tuple[int, int, int] | None = None
            frames = bytearray()
            for chunk in chunks:
                with wave.open(io.BytesIO(chunk), "rb") as wav_file:
                    params = (
                        wav_file.getnchannels(),
                        wav_file.getsampwidth(),
                        wav_file.getframerate(),
                    )
                    if first_params is None:
                        first_params = params
                    elif params != first_params:
                        raise ValueError("synthesized WAV chunks have different formats")
                    frames.extend(wav_file.readframes(wav_file.getnframes()))
            if first_params is not None:
                return _make_wav(bytes(frames), *first_params)
        except (EOFError, OSError, ValueError, wave.Error) as exc:
            print(f"AUDIO: could not merge WAV chunks ({exc}); preserving SDK bytes.")
    audio = b"".join(chunks)
    if audio.startswith(b"RIFF") and b"WAVE" in audio[:16]:
        return audio
    return _make_wav(
        audio,
        OUTPUT_CHANNELS,
        OUTPUT_SAMPLE_WIDTH,
        OUTPUT_SAMPLE_RATE,
    )


def _make_wav(audio: bytes, channels: int, sample_width: int, rate: int) -> bytes:
    import io

    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(audio)
    return stream.getvalue()


def run_session(
    config: AppConfig,
    speechsdk: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    speechsdk = speechsdk or load_speech_sdk()
    try:
        translation_config, auto_detect_config = create_translation_components(
            config, speechsdk
        )
        if config.input_wav is None:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            print("INPUT: default microphone")
        else:
            audio_config = speechsdk.audio.AudioConfig(filename=str(config.input_wav))
            print(f"INPUT: {config.input_wav}")

        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config,
            auto_detect_source_language_config=auto_detect_config,
            audio_config=audio_config,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SessionError(f"Unable to configure Speech session: {exc}") from exc
    collector = AudioCollector(config.output_wav)
    player = LiveAudioPlayer(config.play_audio)
    stopped = threading.Event()
    canceled = {"message": None}

    def on_recognizing(event: Any) -> None:
        print(format_recognizing_event(event))

    def on_recognized(event: Any) -> None:
        print(format_recognized_event(event, speechsdk))

    def on_synthesizing(event: Any) -> None:
        audio = bytes(getattr(event.result, "audio", b"") or b"")
        if audio:
            collector.add(audio)
            player.add(audio)
        print(f"SYNTHESIZING: {len(audio)} byte(s)")

    def on_canceled(event: Any) -> None:
        message = format_canceled_event(event)
        canceled["message"] = message
        stopped.set()

    def on_session_started(_: Any) -> None:
        print("SESSION: started")

    def on_session_stopped(_: Any) -> None:
        print("SESSION: stopped")
        stopped.set()

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.synthesizing.connect(on_synthesizing)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)

    print(f"OUTPUT: {config.target_language}; voice={config.voice_name}")
    print("SESSION: starting (Ctrl+C to stop)")
    try:
        recognizer.start_continuous_recognition()
        deadline = time.monotonic() + config.timeout_seconds
        while not stopped.is_set():
            if time.monotonic() >= deadline:
                print("SESSION: timeout reached")
                break
            sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("SESSION: interrupted")
    except EOFError:
        print("SESSION: EOF received")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SessionError(f"Unable to start or run Speech session: {exc}") from exc
    finally:
        try:
            recognizer.stop_continuous_recognition()
        except (OSError, RuntimeError, ValueError) as exc:
            raise SessionError(f"Unable to stop Speech session: {exc}") from exc
        finish_errors = []
        try:
            collector.finish()
        except (OSError, ValueError, wave.Error) as exc:
            finish_errors.append(f"Unable to write synthesized audio: {exc}")
        try:
            player.finish()
        except SessionError as exc:
            finish_errors.append(str(exc))
        if finish_errors:
            raise SessionError("; ".join(finish_errors))
    if canceled["message"]:
        raise SessionError(canceled["message"])


def main(argv: Iterable[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        config = load_config(args)
        run_session(config)
    except (ConfigurationError, SessionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
