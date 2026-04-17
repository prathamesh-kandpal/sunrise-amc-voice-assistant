from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sunrise_amc_faq.config import AppConfig
from sunrise_amc_faq.pipeline import answer_query, build_vector_store, run_demo
from sunrise_amc_faq.transcribe import save_transcript, transcribe_audio
from sunrise_amc_faq.eval import run_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sunrise AMC voice FAQ assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse the FAQ PDF and build the vector store")
    ingest_parser.add_argument("--no-rebuild", action="store_true", help="Skip rebuilding if the store already exists")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file with Faster-Whisper")
    transcribe_parser.add_argument("--audio-path", default=None, help="Path to the input audio file")
    transcribe_parser.add_argument("--model-size", default=None, help="Faster-Whisper model size")

    ask_parser = subparsers.add_parser("ask", help="Ask a text question against the FAQ")
    ask_parser.add_argument("--query", required=True, help="Investor question to answer")

    run_demo_parser = subparsers.add_parser("run-demo", help="Run the end-to-end assignment flow")
    run_demo_parser.add_argument("--no-rebuild", action="store_true", help="Skip rebuilding the vector index")

    subparsers.add_parser("evaluate", help="Run a lightweight retrieval and citation evaluation")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig()

    if args.command == "ingest":
        store = build_vector_store(config=config, rebuild=not args.no_rebuild)
        print(json.dumps({"collection_count": store.count()}, indent=2))
        return 0

    if args.command == "transcribe":
        audio_path = config.audio_path if args.audio_path is None else Path(args.audio_path).resolve()
        model_size = args.model_size or config.whisper_model_size
        result = transcribe_audio(audio_path=audio_path, model_size=model_size)
        output_path = save_transcript(result, config.transcripts_dir)
        print(json.dumps({"transcript_path": str(output_path), "text": result.text}, indent=2))
        return 0

    if args.command == "ask":
        result = answer_query(config=config, query=args.query)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "run-demo":
        output_path = run_demo(config=config, rebuild_index=not args.no_rebuild)
        print(json.dumps({"response_path": str(output_path)}, indent=2))
        return 0

    if args.command == "evaluate":
        output_path = run_eval(config=config)
        print(json.dumps({"eval_path": str(output_path)}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
