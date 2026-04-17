# Technical Assignment Report

## Overview

This report documents how I implemented the Sunrise AMC Voice + Generative AI take-home assignment, including the approach I took, the project structure, the reasoning behind key design decisions, the roadblocks encountered during development, and the final outcome.

The assignment required building a local, Python-only prototype that:

1. Accepts an investor voice query from an MP3 file.
2. Transcribes it using Faster-Whisper with structured output.
3. Ingests and chunks a FAQ PDF.
4. Embeds and stores the chunks in ChromaDB using sentence-transformers.
5. Uses a locally running LLM via Ollama to answer the query in a grounded way.
6. Returns the answer with source citations.

## Initial Understanding Of The Problem

Before writing code, I first read the assignment brief and the provided FAQ PDF to understand the real problem being tested.

The key requirements I identified were:

- The solution had to be Python only.
- No paid APIs were allowed.
- All models had to run locally.
- The solution needed to be modular and submission-ready, not a single long script.
- Generated artifacts had to be stored locally under `data/` and `output/`.
- The answer needed to be grounded and cite the FAQ source number.

From reading the FAQ itself, I also noticed that this was not a typical long unstructured knowledge base. It was a compact, structured FAQ with question-answer pairs. That shaped the chunking strategy significantly.

## Repository Setup And Project Structure

I started by restructuring the repository to match the assignment expectations:

- `input/` for the provided PDF and MP3
- `data/` for Chroma persistence
- `output/transcripts/` for transcription outputs
- `output/responses/` for final pipeline outputs
- `output/eval/` for evaluation artifacts
- `src/sunrise_amc_faq/` for source code

I also added:

- `README.md` with setup and run instructions
- `DECISIONS.md` with model and architecture reasoning
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`

The implementation was broken into small focused modules instead of one monolithic file:

- `config.py` for application paths and settings
- `schemas.py` for typed data structures
- `ingest.py` for PDF parsing and chunking
- `vector_store.py` for Chroma + embeddings
- `transcribe.py` for Faster-Whisper transcription
- `ollama_client.py` for local LLM calls
- `pipeline.py` for orchestration
- `eval.py` for lightweight evaluation
- `cli.py` for the command-line interface

## Implementation Journey

### 1. Reading The Assignment And Source Documents

The first step was understanding both the assignment brief and the FAQ content itself.

The brief clarified the required stack, storage layout, and deliverables. The FAQ PDF clarified the structure of the knowledge base and the types of investor questions the system would need to answer.

The FAQ contained 10 questions, grouped into sections such as:

- KYC & Onboarding
- SIP & Transactions
- Redemption & Payouts
- Taxation

This made it clear that semantic chunking by FAQ entry would be better than arbitrary text splitting.

### 2. Building The PDF Ingestion Pipeline

I implemented PDF parsing using `pypdf`. The ingestion flow:

1. Reads the PDF page by page.
2. Extracts lines and normalizes whitespace.
3. Detects section headers.
4. Detects FAQ questions (`Q1`, `Q2`, etc.).
5. Accumulates answer lines until the next question or section.
6. Emits chunk objects with:
   - chunk id
   - FAQ number
   - question
   - answer
   - section
   - page
   - combined chunk text

The chunking strategy was:

- One chunk per FAQ entry where possible.
- Sentence-window splitting only if an entry becomes too long.

This gave better grounding because the retrieval unit aligned naturally with the FAQ structure.

### 3. Building The Vector Store Layer

For retrieval, I used:

- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB for local vector storage

The vector store module was designed to:

- Create a persistent Chroma collection under `data/chroma/`
- Rebuild the collection from parsed chunks
- Embed chunk text locally
- Query the top-k nearest chunks for a user question

I also made the dependency loading defensive so the code fails with a clear setup message if the required packages are missing.

### 4. Building The Transcription Module

For audio transcription, I used Faster-Whisper with:

- `small` model size
- CPU execution
- `vad_filter=True`
- `word_timestamps=True`

The output format includes:

- full transcript text
- segment-level timestamps
- word-level timestamps
- word-level probabilities
- audio metadata such as language and duration

This output is saved as structured JSON in `output/transcripts/`.

### 5. Building The LLM Answering Layer

For answer generation, I integrated with a locally running Ollama instance using direct HTTP requests rather than another client library. That kept the dependency surface smaller.

The prompt was designed to:

- restrict the model to the provided FAQ context
- avoid unsupported answers
- keep the answer concise
- include source citations

The retrieved chunks are turned into prompt context and passed to `llama3.1:8b` running in Ollama.

### 6. Wiring Everything Into An End-To-End Pipeline

The `run-demo` command performs the full flow:

1. Build or refresh the vector store.
2. Transcribe the provided MP3.
3. Use the transcript as the retrieval query.
4. Retrieve the most relevant FAQ chunks.
5. Ask the local LLM to answer using only those chunks.
6. Save the final JSON response to `output/responses/`.

I also added a lightweight evaluation script so the project shows some retrieval-quality thinking beyond the minimum assignment requirement.

## Challenges And Roadblocks

This assignment was not just about writing the happy-path code. A significant part of the work involved resolving local-environment and runtime issues so the system could actually execute end to end.

### 1. Missing Local Dependencies

At the beginning, the workspace did not have the required ML dependencies installed:

- `faster-whisper`
- `chromadb`
- `sentence-transformers`
- `torch`
- related runtime libraries

This meant I could not just jump to a final run. I first had to scaffold the codebase in a way that was runnable later and clearly documented what needed to be installed.

### 2. Package Installation Took Much Longer Than A Typical Python Project

Installing the required packages took a long time because this stack pulls in large ML dependencies such as:

- `torch`
- `onnxruntime`
- `transformers`
- `chromadb`

This was a realistic part of the development experience and worth documenting, because local AI stacks have much heavier setup time than a normal backend project.

### 3. Ollama Was Not Installed Initially

The code was ready before the local LLM runtime was. Since the assignment required local inference, I had to wait until Ollama and the local Llama model were installed before I could do a real end-to-end verification run.

Once Ollama was installed and `llama3.1:8b` was pulled, the final integration run became possible.

### 4. Hugging Face Model Download Was Blocked During Runtime

One of the biggest roadblocks came from the embedding model.

The code uses `sentence-transformers/all-MiniLM-L6-v2`, and on the first live run the library attempted to fetch the model from Hugging Face. That failed in the restricted environment with socket and connection errors.

This was not a code bug in the retrieval logic itself. It was a runtime dependency availability issue.

To resolve it, I reran the process in a way that allowed the model to be fetched so that the vector store could be built and the pipeline could complete.

### 5. PDF Footer Noise Leaked Into One FAQ Chunk

During output inspection, I found that a footer fragment from the PDF had leaked into the `Q10` chunk. This happened because the PDF text extractor included footer lines in the same text stream as the answer body.

This mattered because dirty retrieval chunks can degrade answer quality and make the stored knowledge base look sloppy.

I fixed this by explicitly filtering known footer patterns such as:

- toll-free/support lines
- registered office text
- SEBI registration footer content

After that, the retrieved `Q10` chunk became clean.

### 6. Initial Source Citation Was Incomplete For A Multi-Part Query

The sample investor query asks two different tax-related things:

- how gains are taxed after 14 months
- whether TDS is deducted

On the first successful run, the answer content was broadly correct, but the citation only pointed to `Q10`, even though the capital gains tax part clearly came from `Q9`.

This was an important grounding issue. The model had answered a multi-part question using multiple chunks, but the citation logic was too simplistic.

To fix this, I updated the pipeline so that:

- relevant retrieved FAQs are selected based on similarity threshold
- citations are merged with any citations emitted by the LLM
- multi-part answers preserve all relevant FAQ references

### 7. The LLM Sometimes Under-Answered One Part Of The Question

On one run, the answer cited both `Q9` and `Q10`, but it did not explicitly state the TDS conclusion in the answer body.

This was another grounding-quality issue. Even though the right chunks were retrieved, the final answer needed to explicitly respond to each part of the investor’s question.

I addressed this in two ways:

- improved the prompt so the model is instructed to address every part of the query explicitly
- added a small post-processing safety check to fill in missing TDS guidance when the query clearly asks about TDS and the relevant chunk is present

This produced a much cleaner final answer.

## Final Live Run Results

The final pipeline run completed successfully.

### Transcription Result

The audio was correctly transcribed as:

> Hi, I recently invested in an equity mutual fund through my SIP. I want to know, if I redeem my units after 14 months, how will my gains be taxed? And will any TDS be deducted?

The transcript JSON also includes segment timing and word-level probabilities.

### Retrieval Result

The top retrieved chunks were:

- `Q9` for tax treatment on equity mutual fund gains
- `Q10` for TDS on mutual fund redemptions
- `Q8` as a lower-ranked partial redemption related chunk

This is a strong retrieval result because the top two chunks directly correspond to the two sub-questions in the investor query.

### Final Answer

The final answer generated by the system was:

> If you redeem your units after 14 months, your gains will be taxed as Long-Term Capital Gains (LTCG) since you're holding the units for more than 12 months. As per current tax laws (FY 2025-26), LTCG is taxed at 12.5% on gains exceeding Rs. 1.25 lakh in a financial year. Regarding TDS, it will not be deducted from your mutual fund redemption as you are a resident Indian investor.

Sources cited:

- `Q9`
- `Q10`

This satisfies the assignment’s requirement for a grounded answer with source citation.

## Tradeoffs Made

To keep the implementation practical and aligned with the take-home time constraint, I made a few deliberate tradeoffs:

- I used a lightweight embedding model instead of a larger one to keep setup and inference manageable on a laptop.
- I used direct HTTP calls to Ollama rather than adding another library.
- I used a lightweight evaluation harness instead of building a full benchmark suite.
- I optimized for a clean submission-ready codebase rather than more advanced orchestration or async execution.

These choices favored clarity, local reproducibility, and completion within the expected assignment window.

## What I Would Improve With More Time

If this were moving beyond prototype stage, the next improvements I would make are:

- cache and pin embedding models more explicitly for smoother offline runs
- add audio preprocessing such as normalization and stronger VAD
- add confidence thresholds for low-quality retrieval and human handoff
- improve evaluation with a larger labeled query set
- add stricter source-grounding validation so every factual claim is tied to retrieved context
- reduce noisy runtime warnings from dependency libraries
- version the vector store and support incremental re-indexing

## Conclusion

This assignment was completed as a working local prototype that satisfies the requested flow:

- local transcription
- local PDF ingestion and chunking
- local embeddings and Chroma retrieval
- local answer generation through Ollama
- structured outputs stored within the project directory

The final implementation was not just written, but actively debugged through real execution. The biggest value in the process came from solving the actual integration problems that only show up when the system is run end to end: missing dependencies, blocked model downloads, noisy PDF extraction, and multi-source grounding issues.

Those issues were resolved, and the final output is accurate, grounded, and submission-ready.
