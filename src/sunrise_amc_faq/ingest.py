from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from sunrise_amc_faq.schemas import FAQChunk

QUESTION_RE = re.compile(r"^Q(?P<number>\d+)\.\s+(?P<question>.+)$")
SECTION_RE = re.compile(r"^(?P<number>\d+)\.\s+(?P<section>.+)$")
FOOTER_PREFIXES = (
    "Sunrise Asset Management Co. Ltd. |",
    "Mutual Fund investments are subject to market risks.",
)
FOOTER_CONTAINS = (
    "Toll Free:",
    "support@sunriseamc.in",
    "SEBI Reg. No.",
    "Registered Office:",
)


def extract_pdf_lines(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    rows: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            cleaned = " ".join(raw_line.split()).strip()
            if cleaned:
                rows.append((page_index, cleaned))
    return rows


def parse_faq_chunks(pdf_path: Path, chunk_size_chars: int, chunk_overlap_chars: int) -> list[FAQChunk]:
    rows = extract_pdf_lines(pdf_path)
    base_entries: list[FAQChunk] = []

    current_section = "General"
    current_number: int | None = None
    current_question = ""
    current_answer_lines: list[str] = []
    current_page = 1

    def flush_current() -> None:
        nonlocal current_number, current_question, current_answer_lines, current_page
        if current_number is None:
            return
        answer = " ".join(current_answer_lines).strip()
        text = f"FAQ Q{current_number}: {current_question}\nSection: {current_section}\nAnswer: {answer}"
        base_entries.extend(
            split_entry(
                faq_number=current_number,
                question=current_question,
                answer=answer,
                section=current_section,
                page=current_page,
                text=text,
                chunk_size_chars=chunk_size_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
        )
        current_number = None
        current_question = ""
        current_answer_lines = []

    for page, line in rows:
        if line.startswith(FOOTER_PREFIXES):
            continue
        if any(token in line for token in FOOTER_CONTAINS):
            continue

        if line in {"Investor Support Frequently Asked Questions", "Version 2.1 | March 2026 | For Internal Use Only"}:
            continue

        section_match = SECTION_RE.match(line)
        is_section_header = bool(
            section_match
            and section_match.group("section")
            and any(
                token in section_match.group("section").lower()
                for token in ("taxation", "redemption", "sip", "kyc", "onboarding", "transactions", "payouts")
            )
        )
        if is_section_header:
            flush_current()
            current_section = section_match.group("section").strip()
            continue

        question_match = QUESTION_RE.match(line)
        if question_match:
            flush_current()
            current_number = int(question_match.group("number"))
            current_question = question_match.group("question").strip()
            current_answer_lines = []
            current_page = page
            continue

        if current_number is None:
            continue

        current_answer_lines.append(line)

    flush_current()
    return base_entries


def split_entry(
    *,
    faq_number: int,
    question: str,
    answer: str,
    section: str,
    page: int,
    text: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[FAQChunk]:
    if len(text) <= chunk_size_chars:
        return [
            FAQChunk(
                chunk_id=f"faq-{faq_number}-chunk-0",
                faq_number=faq_number,
                question=question,
                answer=answer,
                section=section,
                page=page,
                text=text,
            )
        ]

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    windows: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if candidate and len(candidate) <= chunk_size_chars:
            current = candidate
            continue
        if current:
            windows.append(current)
        if len(sentence) <= chunk_size_chars:
            current = sentence
        else:
            start = 0
            while start < len(sentence):
                end = min(start + chunk_size_chars, len(sentence))
                windows.append(sentence[start:end].strip())
                if end == len(sentence):
                    break
                start = max(end - chunk_overlap_chars, start + 1)
            current = ""
    if current:
        windows.append(current)

    chunks: list[FAQChunk] = []
    for index, window in enumerate(windows):
        chunks.append(
            FAQChunk(
                chunk_id=f"faq-{faq_number}-chunk-{index}",
                faq_number=faq_number,
                question=question,
                answer=answer,
                section=section,
                page=page,
                text=f"FAQ Q{faq_number}: {question}\nSection: {section}\nAnswer: {window}",
            )
        )
    return chunks
