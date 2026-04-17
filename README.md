# Sunrise AMC Voice FAQ Assistant

This project implements the take-home assignment as a local, Python-only pipeline:

1. Transcribe `input/investor_sample.mp3` with Faster-Whisper.
2. Parse and chunk `input/SunriseAMC_FAQ.pdf`.
3. Embed chunks with `sentence-transformers` and store them in ChromaDB.
4. Query a locally running Ollama model for a grounded answer with FAQ citations.

## Project Structure

```text
input/                  Provided PDF and audio input
data/chroma/            Local Chroma persistence
output/transcripts/     Structured transcription JSON
output/responses/       End-to-end pipeline outputs
output/eval/            Retrieval / answer evaluation results
src/sunrise_amc_faq/    Application source code
```

## Setup

1. Create and activate a virtual environment.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

3. Install and start Ollama, then pull a local model:

```bash
ollama serve
ollama pull llama3.1:8b
```

4. Place the provided files in `input/`:
   - `input/SunriseAMC_FAQ.pdf`
   - `input/investor_sample.mp3`

The repository already includes copies of the provided files under `input/`.

## One-Command Run

```bash
python -m sunrise_amc_faq.cli run-demo
```

This command:

1. Builds or refreshes the vector store from the PDF.
2. Transcribes the sample audio.
3. Uses the transcript as the RAG query.
4. Writes the final JSON response to `output/responses/`.

## Useful Commands

```bash
python -m sunrise_amc_faq.cli ingest
python -m sunrise_amc_faq.cli transcribe --audio-path input/investor_sample.mp3
python -m sunrise_amc_faq.cli ask --query "How long does redemption take?"
python -m sunrise_amc_faq.cli evaluate
```

## Notes

- The code expects a local Ollama server at `http://127.0.0.1:11434`.
- If the local LLM is unavailable, the code raises a clear setup error instead of silently falling back to a paid API.
- Generated data stays inside this project under `data/` and `output/`.
