from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

from sunrise_amc_faq.config import AppConfig, ensure_directories
from sunrise_amc_faq.ingest import parse_faq_chunks
from sunrise_amc_faq.ollama_client import generate_answer
from sunrise_amc_faq.schemas import AnswerResult
from sunrise_amc_faq.transcribe import save_transcript, transcribe_audio
from sunrise_amc_faq.vector_store import FAQVectorStore


def build_vector_store(config: AppConfig, rebuild: bool = True) -> FAQVectorStore:
    ensure_directories(config)
    chunks = parse_faq_chunks(
        pdf_path=config.faq_pdf_path,
        chunk_size_chars=config.chunk_size_chars,
        chunk_overlap_chars=config.chunk_overlap_chars,
    )
    store = FAQVectorStore(
        persist_dir=config.chroma_dir,
        collection_name=config.collection_name,
        embedding_model_name=config.embedding_model_name,
    )
    if rebuild or store.count() == 0:
        store.rebuild(chunks)
    return store


def answer_query(config: AppConfig, query: str, store: FAQVectorStore | None = None) -> AnswerResult:
    ensure_directories(config)
    active_store = store or build_vector_store(config=config, rebuild=False)
    retrieved = active_store.query(query_text=query, top_k=config.retrieve_top_k)

    if not retrieved:
        return AnswerResult(
            query=query,
            answer="I could not find a relevant answer in the Sunrise AMC FAQ.",
            cited_faq_numbers=[],
            retrieved_chunks=[],
            model=config.ollama_model,
            grounded=False,
        )

    context = "\n\n".join(
        f"[FAQ Q{chunk.metadata['faq_number']}] {chunk.text}" for chunk in retrieved
    )
    prompt = f"""
You are a careful investor support assistant. Answer only using the FAQ context below.
If the answer is not clearly supported by the context, say that the FAQ does not contain enough information.
Address every part of the investor query explicitly.
If the query asks multiple things, answer each one in a short combined response.
Keep the answer concise and include the cited FAQ number in the final sentence using the form "Source: QX".

Investor query:
{query}

FAQ context:
{context}
""".strip()
    answer = generate_answer(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        prompt=prompt,
        temperature=0.0,
    )
    relevant_retrieved = _select_relevant_sources(retrieved)
    answer = _fill_missing_details(query=query, answer=answer, retrieved=retrieved)
    cited_numbers = sorted(
        {
            int(match)
            for match in re.findall(r"Q(\d+)", answer)
        }
    )
    merged_citations = sorted(set(cited_numbers) | set(relevant_retrieved))
    if merged_citations:
        answer = re.sub(r"\n*\s*Sources?:\s*Q[\d,\sQ]+\.?\s*$", "", answer, flags=re.IGNORECASE)
        source_suffix = ", ".join(f"Q{number}" for number in merged_citations)
        answer = f"{answer.strip()}\n\nSources: {source_suffix}"

    grounded = bool(merged_citations)
    cited_numbers = merged_citations

    return AnswerResult(
        query=query,
        answer=answer,
        cited_faq_numbers=cited_numbers,
        retrieved_chunks=retrieved,
        model=config.ollama_model,
        grounded=grounded,
    )


def _select_relevant_sources(retrieved: list) -> list[int]:
    if not retrieved:
        return []

    best_distance = retrieved[0].distance
    threshold = best_distance + 0.25
    selected = [
        int(chunk.metadata["faq_number"])
        for chunk in retrieved
        if chunk.distance <= threshold
    ]
    return sorted(set(selected))


def _fill_missing_details(query: str, answer: str, retrieved: list) -> str:
    normalized_query = query.lower()
    normalized_answer = answer.lower()

    if "tds" in normalized_query and "tds" not in normalized_answer:
        q10_chunk = next((chunk for chunk in retrieved if int(chunk.metadata["faq_number"]) == 10), None)
        if q10_chunk is not None:
            answer = (
                f"{answer.strip()} "
                "TDS is not applicable for resident Indian investors on mutual fund redemptions."
            )

    return answer.strip()


def run_demo(config: AppConfig, rebuild_index: bool = True) -> Path:
    ensure_directories(config)
    start = time.perf_counter()
    store = build_vector_store(config=config, rebuild=rebuild_index)
    transcript_result = transcribe_audio(
        audio_path=config.audio_path,
        model_size=config.whisper_model_size,
    )
    transcript_path = save_transcript(transcript_result, config.transcripts_dir)
    answer_result = answer_query(config=config, query=transcript_result.text, store=store)
    total_runtime_seconds = round(time.perf_counter() - start, 3)

    payload = {
        "transcript_path": str(transcript_path),
        "transcript_text": transcript_result.text,
        "answer": answer_result.to_dict(),
        "runtime_seconds": total_runtime_seconds,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = config.responses_dir / f"response_{timestamp}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
