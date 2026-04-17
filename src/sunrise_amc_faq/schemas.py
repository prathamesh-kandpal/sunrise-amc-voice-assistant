from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FAQChunk:
    chunk_id: str
    faq_number: int
    question: str
    answer: str
    section: str
    page: int
    text: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "faq_number": self.faq_number,
            "question": self.question,
            "section": self.section,
            "page": self.page,
        }


@dataclass
class TranscriptWord:
    word: str
    start: float
    end: float
    probability: float | None


@dataclass
class TranscriptSegment:
    segment_id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None
    no_speech_prob: float | None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass
class TranscriptResult:
    audio_path: str
    model: str
    language: str | None
    duration_seconds: float | None
    text: str
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk_id: str
    distance: float
    text: str
    metadata: dict[str, Any]


@dataclass
class AnswerResult:
    query: str
    answer: str
    cited_faq_numbers: list[int]
    retrieved_chunks: list[RetrievedChunk]
    model: str
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
