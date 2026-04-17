from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sunrise_amc_faq.config import AppConfig, ensure_directories
from sunrise_amc_faq.pipeline import answer_query, build_vector_store


@dataclass
class EvalCase:
    query: str
    expected_faq_number: int


def default_eval_cases() -> list[EvalCase]:
    return [
        EvalCase("How long does KYC verification take?", 2),
        EvalCase("Can I pause my SIP for a few months?", 5),
        EvalCase("What happens if my SIP payment fails twice?", 6),
        EvalCase("How long will redemption from an equity fund take?", 7),
        EvalCase("Is TDS deducted for resident investors during redemption?", 10),
    ]


def run_eval(config: AppConfig) -> Path:
    ensure_directories(config)
    store = build_vector_store(config=config, rebuild=False)
    cases = default_eval_cases()
    results: list[dict] = []
    top1_hits = 0
    answer_hits = 0

    for case in cases:
        answer = answer_query(config=config, query=case.query, store=store)
        top1 = answer.retrieved_chunks[0].metadata["faq_number"] if answer.retrieved_chunks else None
        top1_correct = top1 == case.expected_faq_number
        cited_correct = case.expected_faq_number in answer.cited_faq_numbers
        top1_hits += int(top1_correct)
        answer_hits += int(cited_correct)
        results.append(
            {
                **asdict(case),
                "top1_faq_number": top1,
                "top1_correct": top1_correct,
                "cited_faq_numbers": answer.cited_faq_numbers,
                "answer_cited_expected_faq": cited_correct,
                "answer_preview": answer.answer,
            }
        )

    payload = {
        "cases": results,
        "retrieval_top1_accuracy": round(top1_hits / len(cases), 3),
        "citation_hit_rate": round(answer_hits / len(cases), 3),
    }
    destination = config.eval_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
