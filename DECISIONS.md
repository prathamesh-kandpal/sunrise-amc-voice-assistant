# Decisions

## Model Selection

- LLM: `llama3.1:8b` via Ollama
  - Good quality-to-latency tradeoff for a laptop or single-GPU environment.
  - Strong enough for short grounded FAQ answering without requiring a larger model footprint.
  - Easy local setup through Ollama, which keeps the solution compliant with the assignment constraints.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - Small and fast.
  - Well-suited to short FAQ passages and semantic retrieval on commodity hardware.
  - Lower latency matters more here than squeezing out marginal benchmark improvements.
- ASR model: `faster-whisper` with `small` by default
  - Fast enough for short support calls.
  - Better latency than larger Whisper checkpoints while preserving solid quality for clear English speech.

## Chunking Strategy

The FAQ is not a general document corpus; it is a structured list of question-and-answer entries. Because of that, the chunking strategy uses each FAQ entry as the primary semantic unit:

- One chunk per FAQ question-answer pair whenever the full entry fits within the target chunk size.
- The section name and FAQ number are attached as metadata.
- If an entry is unusually long, it is split into overlapping sentence windows while preserving the same FAQ metadata.

Why this is better than naive character splitting:

- Retrieval aligns with how users ask support questions.
- The answer remains grounded to a single FAQ number whenever possible.
- It avoids breaking policy conditions or timelines across arbitrary chunk boundaries.

For this PDF, most answers are short enough that one FAQ pair naturally maps to one chunk, which is the cleanest and least lossy strategy.

## Tradeoffs Made

- I optimized for clarity and reproducibility over maximum retrieval sophistication.
- I used a lightweight embedding model instead of a larger BGE or E5 model to keep local setup simple.
- I used direct Ollama HTTP calls instead of adding another client dependency.
- The evaluation harness is intentionally lightweight and focused on retrieval correctness and citation behavior, not full generative grading.

## Production Readiness

For a production system, I would change the following:

- Add VAD, audio normalization, and language detection before transcription.
- Add retry logic, timeouts, and structured observability around Ollama and vector DB operations.
- Version the vector index and store document fingerprints to support safe re-indexing.
- Add confidence thresholds and route low-confidence or out-of-scope queries to a human agent.
- Add a stronger offline evaluation suite with labeled support queries and citation accuracy checks.
- Use asynchronous workers and request queues for transcription and inference to improve throughput.
- Add PII-aware logging and redaction because investor support queries may contain account-sensitive information.

## What Would Not Scale

- Rebuilding the entire Chroma collection for every document change would not scale.
- A single local Ollama instance on one laptop would become a bottleneck under concurrent support traffic.
- The current prompt-only grounding is good for a prototype, but production would benefit from stricter answer templating and citation verification.

## Environment Limitation

This workspace did not have the required local inference dependencies preinstalled during development. The codebase therefore includes explicit setup instructions and defensive runtime checks so the project remains runnable once the required open-source packages and a local Ollama model are installed.
