"""Chroma-backed vector store over report text chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from esg_extractor.ingestion.pdf_parser import TextChunk
from esg_extractor.rag.embeddings import Embedder


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_start: int
    page_end: int
    distance: float


class ReportVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "esg_reports",
        embedder: Optional[Embedder] = None,
    ):
        import chromadb

        self.embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_file": c.source_file, "page_start": c.page_start, "page_end": c.page_end}
                for c in chunks
            ],
        )

    def query(self, question: str, top_k: int = 5, source_file: Optional[str] = None) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_one(question)
        where = {"source_file": source_file} if source_file else None
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
        chunks: list[RetrievedChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source_file=meta["source_file"],
                    page_start=meta["page_start"],
                    page_end=meta["page_end"],
                    distance=dist,
                )
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()
