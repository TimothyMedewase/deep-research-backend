from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import chromadb
from dotenv import load_dotenv

from agent.memory.vector_store import VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate local Chroma collections into Chroma Cloud."
    )
    parser.add_argument(
        "--persist-dir",
        default=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        help="Path to the local PersistentClient directory.",
    )
    parser.add_argument(
        "--collection-prefix",
        default=os.getenv("CHROMA_COLLECTION_PREFIX", "research-memory"),
        help="Prefix for destination Chroma Cloud collections.",
    )
    parser.add_argument(
        "--default-scope",
        default=os.getenv("CHROMA_DEFAULT_SCOPE", ""),
        help="Fallback memory scope when records do not already have one.",
    )
    parser.add_argument(
        "--only-collection",
        default="",
        help="Migrate just one source collection by exact name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many source records to read per batch.",
    )
    return parser.parse_args()


def _require_cloud_env() -> None:
    missing = [
        name
        for name in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise SystemExit(
            "Missing required Chroma Cloud env vars for migration: "
            + ", ".join(missing)
        )


def _infer_session_id(collection_name: str, metadata: dict[str, Any]) -> str:
    session_id = str(metadata.get("session_id") or "").strip()
    if session_id:
        return session_id
    if collection_name.startswith("session_"):
        return collection_name.removeprefix("session_")
    return collection_name


async def run() -> None:
    load_dotenv()
    args = parse_args()
    _require_cloud_env()

    source_client = chromadb.PersistentClient(path=args.persist_dir)
    collections = list(source_client.list_collections())
    if args.only_collection:
        collections = [c for c in collections if c.name == args.only_collection]

    if not collections:
        raise SystemExit("No source collections found to migrate.")

    stores: dict[tuple[str, str], VectorStore] = {}
    migrated = 0

    for collection in collections:
        offset = 0
        while True:
            batch = collection.get(
                limit=args.batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch.get("ids", [])
            documents = batch.get("documents") or []
            metadatas = batch.get("metadatas") or []
            if not ids:
                break

            for document, metadata in zip(documents, metadatas):
                if not document:
                    continue
                row_metadata = dict(metadata or {})
                session_id = _infer_session_id(collection.name, row_metadata)
                memory_scope = (
                    str(row_metadata.get("memory_scope") or "").strip()
                    or args.default_scope.strip()
                    or session_id
                )
                key = (session_id, memory_scope)
                store = stores.get(key)
                if store is None:
                    store = VectorStore(
                        session_id=session_id,
                        persist_dir=None,
                        chroma_host=os.getenv("CHROMA_HOST", "").strip() or None,
                        chroma_api_key=os.getenv("CHROMA_API_KEY", "").strip() or None,
                        chroma_tenant=os.getenv("CHROMA_TENANT", "").strip() or None,
                        chroma_database=os.getenv("CHROMA_DATABASE", "").strip() or None,
                        collection_prefix=args.collection_prefix,
                        memory_scope=memory_scope,
                    )
                    stores[key] = store

                await store.add(str(document), row_metadata)
                migrated += 1

            offset += len(ids)

    print(
        f"Migrated {migrated} records from {len(collections)} local collections "
        f"into {len(stores)} Chroma Cloud shard(s)."
    )


if __name__ == "__main__":
    asyncio.run(run())
