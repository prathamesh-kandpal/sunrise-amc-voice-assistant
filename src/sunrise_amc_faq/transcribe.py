from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sunrise_amc_faq.schemas import TranscriptResult, TranscriptSegment, TranscriptWord


class MissingTranscriptionDependencyError(RuntimeError):
    """Raised when Faster-Whisper is not installed."""


def _load_whisper_model_class():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MissingTranscriptionDependencyError(
            "Missing Faster-Whisper. Install dependencies with "
            "`pip install -r requirements.txt && pip install -e .`."
        ) from exc
    return WhisperModel


def transcribe_audio(audio_path: Path, model_size: str, compute_type: str = "int8") -> TranscriptResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    whisper_model_cls = _load_whisper_model_class()
    model = whisper_model_cls(model_size, device="cpu", compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    segment_payload: list[TranscriptSegment] = []
    full_text_parts: list[str] = []
    duration_seconds = None

    for index, segment in enumerate(segments):
        full_text_parts.append(segment.text.strip())
        words = [
            TranscriptWord(
                word=word.word,
                start=round(word.start, 3),
                end=round(word.end, 3),
                probability=round(word.probability, 4) if word.probability is not None else None,
            )
            for word in (segment.words or [])
        ]
        segment_payload.append(
            TranscriptSegment(
                segment_id=index,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                text=segment.text.strip(),
                avg_logprob=round(segment.avg_logprob, 4) if segment.avg_logprob is not None else None,
                no_speech_prob=round(segment.no_speech_prob, 4) if segment.no_speech_prob is not None else None,
                words=words,
            )
        )
        duration_seconds = max(duration_seconds or 0.0, segment.end)

    return TranscriptResult(
        audio_path=str(audio_path),
        model=model_size,
        language=getattr(info, "language", None),
        duration_seconds=round(duration_seconds, 3) if duration_seconds is not None else None,
        text=" ".join(part for part in full_text_parts if part).strip(),
        segments=segment_payload,
    )


def save_transcript(result: TranscriptResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"transcript_{timestamp}.json"
    destination.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return destination
