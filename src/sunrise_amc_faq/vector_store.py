from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sunrise_amc_faq.schemas import FAQChunk, RetrievedChunk


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is not installed."""


def _load_runtime_dependencies():
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise MissingDependencyError(
            "Missing vector-store dependencies. Install requirements with "
            "`pip install -r requirements.txt && pip install -e .`."
        ) from exc
    return chromadb, SentenceTransformer


class FAQVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_model_name: str) -> None:
        chromadb, sentence_transformers_cls = _load_runtime_dependencies()
        self._embedding_model = sentence_transformers_cls(embedding_model_name)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def rebuild(self, chunks: Iterable[FAQChunk]) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)

        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("No chunks were generated from the FAQ PDF.")

        texts = [chunk.text for chunk in chunk_list]
        embeddings = self._embedding_model.encode(texts, normalize_embeddings=True).tolist()
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunk_list],
            documents=texts,
            embeddings=embeddings,
            metadatas=[chunk.to_metadata() for chunk in chunk_list],
        )

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self._embedding_model.encode([query_text], normalize_embeddings=True).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        payload: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            payload.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    distance=float(distance),
                    text=document,
                    metadata=dict(metadata),
                )
            )
        return payload
