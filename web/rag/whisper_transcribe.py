"""
Local Whisper transcription. Supports either:
  - faster-whisper (recommended): pip install faster-whisper
  - openai-whisper: pip install openai-whisper

Configure with WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE (faster-whisper only).
WHISPER_VAD_FILTER=1 can drop whole browser recordings; default is off.
WHISPER_BEAM_SIZE default 1 (fast); raise to 5 for slightly better quality, slower.
WHISPER_CONDITION_ON_PREVIOUS 0|1 — default 0 for speed (faster-whisper only).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_model = None
_backend: str | None = None
_lock = threading.Lock()


def detect_backend() -> str | None:
    try:
        import faster_whisper  # noqa: F401

        return "faster"
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401

        return "openai"
    except ImportError:
        return None


def _device_for_faster() -> str:
    raw = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
    if raw in ("cuda", "cpu"):
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _compute_type_for_faster(device: str) -> str:
    env = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip()
    if env:
        return env
    return "float16" if device == "cuda" else "int8"


def _load_model_faster():
    from faster_whisper import WhisperModel

    model_size = os.environ.get("WHISPER_MODEL", "small").strip() or "small"
    device = _device_for_faster()
    compute_type = _compute_type_for_faster(device)
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _load_model_openai():
    import whisper

    model_size = os.environ.get("WHISPER_MODEL", "small").strip() or "small"
    device = _device_for_faster()
    if device == "cuda":
        return whisper.load_model(model_size, device="cuda")
    return whisper.load_model(model_size, device="cpu")


def _get_model():
    global _model, _backend
    with _lock:
        if _model is not None:
            return _model, _backend
        _backend = detect_backend()
        if _backend is None:
            raise RuntimeError(
                "No Whisper package found. Install one of: pip install faster-whisper  OR  pip install openai-whisper"
            )
        if _backend == "faster":
            _model = _load_model_faster()
        else:
            _model = _load_model_openai()
        print(f"[whisper] loaded backend={_backend} model={os.environ.get('WHISPER_MODEL', 'small')!r}")
        return _model, _backend


def transcribe_file(path: Path, language: str | None) -> tuple[str, str]:
    """
    Transcribe audio file. language: ISO code e.g. en, it, or None for auto.
    Returns (text, backend_name).
    """
    model, backend = _get_model()
    lang = (language or "").strip() or None
    if lang and len(lang) > 2:
        lang = lang[:2]

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    if backend == "faster":
        vad = os.environ.get("WHISPER_VAD_FILTER", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            beam = max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "1")))
        except ValueError:
            beam = 1
        cond_prev = os.environ.get("WHISPER_CONDITION_ON_PREVIOUS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        segments, _info = model.transcribe(
            str(path),
            language=lang,
            beam_size=beam,
            vad_filter=vad,
            condition_on_previous_text=cond_prev,
            without_timestamps=True,
        )
        text = "".join(s.text for s in segments).strip()
        return text, "faster-whisper"

    result = model.transcribe(str(path), language=lang)
    text = (result.get("text") or "").strip()
    return text, "openai-whisper"
