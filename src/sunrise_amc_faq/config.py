from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT_DIR / "input"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"


@dataclass(frozen=True)
class AppConfig:
    faq_pdf_path: Path = INPUT_DIR / "SunriseAMC_FAQ.pdf"
    audio_path: Path = INPUT_DIR / "investor_sample.mp3"
    chroma_dir: Path = DATA_DIR / "chroma"
    transcripts_dir: Path = OUTPUT_DIR / "transcripts"
    responses_dir: Path = OUTPUT_DIR / "responses"
    eval_dir: Path = OUTPUT_DIR / "eval"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    whisper_model_size: str = "small"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    collection_name: str = "sunrise_amc_faq"
    retrieve_top_k: int = 3
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 120


def ensure_directories(config: AppConfig) -> None:
    for path in (
        INPUT_DIR,
        DATA_DIR,
        OUTPUT_DIR,
        config.chroma_dir,
        config.transcripts_dir,
        config.responses_dir,
        config.eval_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
