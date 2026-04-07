from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlparse

import chromadb
from chromadb import K, Knn, Rrf, Schema, Search, SparseVectorIndexConfig, VectorIndexConfig
from chromadb.execution.expression.operator import GroupBy, MinK
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import (
    ChromaCloudQwenEmbeddingModel,
)
from chromadb.utils.embedding_functions.chroma_cloud_splade_embedding_function import (
    ChromaCloudSpladeEmbeddingModel,
)
from openai import AsyncOpenAI

from agent.constraints import ConstraintTracker
from agent.json_utils import make_json_safe

MAX_DOCUMENT_BYTES = 14 * 1024


class VectorStoreError(Exception):
    pass


class VectorStore:
    def __init__(
        self,
        session_id: str,
        persist_dir: str | None,
        *,
        chroma_url: str | None = None,
        chroma_host: str | None = None,
        chroma_api_key: str | None = None,
        chroma_tenant: str | None = None,
        chroma_database: str | None = None,
        memory_scope: str | None = None,
        collection_name: str | None = None,
        collection_prefix: str = "research-memory",
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        self.session_id = session_id
        self.persist_dir = persist_dir
        self.chroma_url = (chroma_url or "").strip() or None
        self.chroma_host = (chroma_host or "").strip() or "api.trychroma.com"
        self.chroma_api_key = (chroma_api_key or "").strip() or None
        self.chroma_tenant = (chroma_tenant or "").strip() or None
        self.chroma_database = (chroma_database or "").strip() or None
        self.memory_scope = _normalize_scope(memory_scope or "default")
        self.collection_name = (
            (collection_name or "").strip()
            or _build_collection_name(collection_prefix, self.memory_scope)
        )
        self.openai_client = openai_client or AsyncOpenAI()
        self.mode = self._resolve_mode()
        self.cleanup_on_close = self.mode == "local"
        self.rrf_k = 60
        self.hybrid_dense_weight = 0.75
        self.hybrid_sparse_weight = 0.25
        self.dense_embedding_function: ChromaCloudQwenEmbeddingFunction | None = None
        self.sparse_embedding_function: ChromaCloudSpladeEmbeddingFunction | None = None

        try:
            self.client = self._build_client()
            self.collection = self._build_collection()
        except Exception as exc:
            raise VectorStoreError("Failed to initialize vector store.") from exc

    def _resolve_mode(self) -> str:
        cloud_fields = [
            self.chroma_api_key,
            self.chroma_tenant,
            self.chroma_database,
        ]
        if any(cloud_fields):
            if not all(cloud_fields):
                missing = [
                    name
                    for name, value in (
                        ("CHROMA_API_KEY", self.chroma_api_key),
                        ("CHROMA_TENANT", self.chroma_tenant),
                        ("CHROMA_DATABASE", self.chroma_database),
                    )
                    if not value
                ]
                raise VectorStoreError(
                    "Incomplete Chroma Cloud config. Missing: "
                    + ", ".join(missing)
                )
            return "cloud"
        if self.chroma_url:
            return "server"
        return "local"

    def _build_client(self) -> Any:
        if self.mode == "cloud":
            return chromadb.CloudClient(
                tenant=self.chroma_tenant,
                database=self.chroma_database,
                api_key=self.chroma_api_key,
                cloud_host=self.chroma_host,
            )
        if self.mode == "server":
            parsed = urlparse(self.chroma_url or "")
            if not parsed.scheme or not parsed.hostname:
                raise VectorStoreError(
                    "CHROMA_URL must be a full URL such as http://localhost:8000."
                )
            return chromadb.HttpClient(
                host=parsed.hostname,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
                ssl=parsed.scheme == "https",
            )
        if not self.persist_dir:
            raise VectorStoreError(
                "CHROMA_PERSIST_DIR is required when Chroma Cloud is unset."
            )
        return chromadb.PersistentClient(path=self.persist_dir)

    def _build_collection(self) -> Any:
        metadata = {
            "description": "Deep research session memory",
            "memory_scope": self.memory_scope,
        }
        if self.mode == "cloud":
            dense_ef = ChromaCloudQwenEmbeddingFunction(
                model=ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B,
                task=None,
            )
            sparse_ef = ChromaCloudSpladeEmbeddingFunction(
                model=ChromaCloudSpladeEmbeddingModel.SPLADE_PP_EN_V1
            )
            self.dense_embedding_function = dense_ef
            self.sparse_embedding_function = sparse_ef
            schema = Schema()
            schema.create_index(
                config=VectorIndexConfig(embedding_function=dense_ef)
            )
            schema.create_index(
                config=SparseVectorIndexConfig(
                    source_key=K.DOCUMENT,
                    embedding_function=sparse_ef,
                ),
                key="sparse_embedding",
            )
            self.client.get_or_create_collection(
                name=self.collection_name,
                schema=schema,
                metadata=metadata,
                embedding_function=None,
            )
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=dense_ef,
            )
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
        )

    async def add(self, text: str, metadata: dict) -> int:
        normalized = self._normalize_metadata(metadata, text)
        chunks = _chunk_text(text, max_bytes=MAX_DOCUMENT_BYTES)
        if not chunks:
            return 0
        ids = [_record_id(normalized, chunk, idx) for idx, chunk in enumerate(chunks)]
        metadatas = [
            {
                **normalized,
                "chunk_index": idx,
                "chunk_count": len(chunks),
            }
            for idx in range(len(chunks))
        ]

        try:
            if self.mode == "cloud":
                await asyncio.to_thread(
                    self.collection.upsert,
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas,
                )
                return 0

            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=chunks,
            )
            embeddings = [row.embedding for row in response.data]
            token_count = getattr(
                response.usage,
                "prompt_tokens",
                sum(ConstraintTracker.estimate_tokens(chunk) for chunk in chunks),
            )
            await asyncio.to_thread(
                self.collection.upsert,
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return token_count
        except Exception as exc:
            raise VectorStoreError("Failed to add document to vector store.") from exc

    async def query(self, text: str, n_results: int = 3) -> list[dict]:
        if not text.strip():
            return []
        if self.mode == "cloud":
            return await self._query_cloud(text, n_results)
        return await self._query_legacy(text, n_results)

    async def _query_legacy(self, text: str, n_results: int) -> list[dict]:
        limit = max(n_results * 4, 12)
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            query_embedding = response.data[0].embedding
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"session_id": self.session_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError("Failed to query vector store.") from exc

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        hits = [
            {
                "text": document or "",
                "metadata": metadata or {},
                "distance": float(distance) if distance is not None else 1.0,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
            if document
        ]
        return _dedupe_hits(hits, limit=n_results)

    async def _query_cloud(self, text: str, n_results: int) -> list[dict]:
        limit = max(n_results * 4, 12)
        dense_ef = self.dense_embedding_function
        sparse_ef = self.sparse_embedding_function
        if dense_ef is None or sparse_ef is None:
            raise VectorStoreError("Chroma Cloud embedding functions are not initialized.")

        try:
            dense_query = await asyncio.to_thread(dense_ef.embed_query, [text])
            sparse_query = await asyncio.to_thread(sparse_ef, [text])
            rank_expr = Rrf(
                ranks=[
                    Knn(
                        query=dense_query[0].tolist(),
                        limit=limit,
                        return_rank=True,
                    ),
                    Knn(
                        query=sparse_query[0],
                        key="sparse_embedding",
                        limit=limit,
                        return_rank=True,
                    ),
                ],
                weights=[self.hybrid_dense_weight, self.hybrid_sparse_weight],
                k=self.rrf_k,
                normalize=True,
            )
            search = (
                Search()
                .rank(rank_expr)
                .where({"session_id": self.session_id})
                .group_by(
                    GroupBy(
                        keys=K("source_document_id"),
                        aggregate=MinK(keys=K.SCORE, k=1),
                    )
                )
                .limit(n_results)
                .select(K.ID, K.DOCUMENT, K.METADATA, K.SCORE)
            )
            results = await asyncio.to_thread(self.collection.search, search)
            rows = results.rows()[0]
        except Exception:
            try:
                dense_query = await asyncio.to_thread(dense_ef.embed_query, [text])
                fallback = (
                    Search()
                    .rank(Knn(query=dense_query[0].tolist(), limit=limit))
                    .where({"session_id": self.session_id})
                    .limit(limit)
                    .select(K.ID, K.DOCUMENT, K.METADATA, K.SCORE)
                )
                results = await asyncio.to_thread(self.collection.search, fallback)
                rows = results.rows()[0]
            except Exception as exc:
                raise VectorStoreError("Failed to query vector store.") from exc

        hits = [
            {
                "text": row.get("document") or "",
                "metadata": make_json_safe(row.get("metadata") or {}),
                "distance": float(row.get("score")) if row.get("score") is not None else 1.0,
            }
            for row in rows
            if row.get("document")
        ]
        return _dedupe_hits(hits, limit=n_results)

    def delete_collection(self) -> None:
        if not self.cleanup_on_close:
            return
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as exc:
            raise VectorStoreError("Failed to delete vector store collection.") from exc

    def _normalize_metadata(self, metadata: dict[str, Any], text: str) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in (metadata or {}).items():
            normalized[str(key)] = _normalize_metadata_value(value)

        normalized["session_id"] = self.session_id
        normalized["memory_scope"] = self.memory_scope
        normalized.setdefault("document_kind", _document_kind(normalized))
        normalized.setdefault("source_document_id", _source_document_id(normalized, text))
        return normalized


def _normalize_scope(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    return normalized or "default"


def _build_collection_name(prefix: str, scope: str) -> str:
    prefix_norm = _normalize_scope(prefix or "research-memory")
    scope_norm = _normalize_scope(scope)
    name = f"{prefix_norm}-{scope_norm}"
    if len(name) <= 63:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[:54]}-{digest}"


def _normalize_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _document_kind(metadata: dict[str, Any]) -> str:
    if metadata.get("url"):
        return "document"
    if metadata.get("source"):
        return str(metadata["source"])
    return "note"


def _source_document_id(metadata: dict[str, Any], text: str) -> str:
    if metadata.get("source_document_id"):
        return str(metadata["source_document_id"])
    for key in ("url", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"doc-{digest}"


def _record_id(metadata: dict[str, Any], chunk: str, chunk_index: int) -> str:
    payload = "|".join(
        [
            str(metadata.get("memory_scope", "")),
            str(metadata.get("session_id", "")),
            str(metadata.get("source_document_id", "")),
            str(chunk_index),
            chunk,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _chunk_text(text: str, *, max_bytes: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped.encode("utf-8")) <= max_bytes:
        return [stripped]

    lines = stripped.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        line = line.rstrip()
        if not line:
            candidate = "\n".join([*current, ""]).strip()
            if candidate and len(candidate.encode("utf-8")) <= max_bytes:
                current.append("")
                continue
            flush()
            continue

        if len(line.encode("utf-8")) > max_bytes:
            flush()
            chunks.extend(_split_long_line(line, max_bytes=max_bytes))
            continue

        candidate = "\n".join([*current, line]).strip()
        if candidate and len(candidate.encode("utf-8")) > max_bytes:
            flush()
        current.append(line)

    flush()
    return [chunk for chunk in chunks if chunk]


def _split_long_line(line: str, *, max_bytes: int) -> list[str]:
    words = line.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if candidate and len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(word.encode("utf-8")) <= max_bytes:
            current = word
            continue

        start = 0
        while start < len(word):
            end = start + max(1, max_bytes // 2)
            piece = word[start:end]
            while piece and len(piece.encode("utf-8")) > max_bytes:
                end -= 1
                piece = word[start:end]
            if not piece:
                break
            chunks.append(piece)
            start = end

    if current:
        chunks.append(current)
    return chunks


def _dedupe_hits(hits: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        metadata = hit.get("metadata") or {}
        source_document_id = str(metadata.get("source_document_id") or "")
        dedupe_key = source_document_id or hashlib.sha1(
            (hit.get("text") or "").encode("utf-8")
        ).hexdigest()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(hit)
        if len(out) >= limit:
            break
    return out
