# n8n / Dify integration

The FastAPI service stays the runtime for research; **n8n** or **Dify** plug in via HTTPS webhooks so you can route queries and own long-term memory without forking Python.

**By default this is required:** `N8N_INTEGRATION_REQUIRED=true` (the default) makes the app **fail at startup** unless `QUERY_ROUTER_WEBHOOK_URL` and `EXTERNAL_MEMORY_WEBHOOK_URL` are set. For local development without workflows, set `N8N_INTEGRATION_REQUIRED=false` in `.env`.

## 1. Query router (`QUERY_ROUTER_WEBHOOK_URL`)

**POST** JSON body:

```json
{ "query": "user text", "session_id": "uuid" }
```

Respond with:

```json
{
  "route": "deep_research",
  "query": "optional rewritten query",
  "constraints": { "max_sub_questions": 3, "max_cost_usd": 0.05, "max_tokens": 6000 }
}
```

- `route`: `deep_research` | `memory_only` | `blocked`
- Optional `QUERY_ROUTER_WEBHOOK_BEARER` sends `Authorization: Bearer …`

**n8n:** Webhook node → your logic → Respond to Webhook.  
**Dify:** HTTP API / custom tool that returns the JSON above.

## 2. External memory (`EXTERNAL_MEMORY_WEBHOOK_URL`)

One URL can handle three actions; branch on `action` (Switch node in n8n).

| `action` | Body fields | Suggested response |
|----------|-------------|-------------------|
| `retrieve_similar` | `query`, `session_id` | `{ "hits": [ { "text", "metadata", "distance"? } ] }` or `chunks` / `records` |
| `retrieve_documents` | `query`, `session_id` | `{ "results": [ { "url", "title", "content" } ] }` (used in `memory_only` + tool `fetch_external_documents`) |
| `store` | `session_id`, `text`, `metadata` | any JSON, status 2xx |

Optional `EXTERNAL_MEMORY_STORE_WEBHOOK_URL` overrides the URL for `store` only.  
Optional `EXTERNAL_MEMORY_WEBHOOK_BEARER`.

### Chroma Cloud search

The backend now prefers **Chroma Cloud** over a local Chroma server when these env vars are set:

```bash
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=...
CHROMA_TENANT=25b8f9e7-2c7b-4dc5-8202-d077e96396db
CHROMA_DATABASE=reaserch-agent
CHROMA_COLLECTION_PREFIX=research-memory
CHROMA_DEFAULT_SCOPE=default
```

Behavior:

- The app uses `chromadb.CloudClient(...)` and `get_or_create_collection(...)`.
- Each collection is named from `CHROMA_COLLECTION_PREFIX` plus a scope key.
- Pass `memory_scope` in `POST /research` to shard mutually exclusive data by user or organization.
- Retrieval still filters by `session_id`, so the in-app memory tool remains session-scoped even inside a larger shard.
- Documents are chunked before storage to stay under Chroma’s 16 KiB record limit.
- Each chunk stores `source_document_id`, `chunk_index`, and `chunk_count` metadata so hybrid retrieval can deduplicate by source.

Hybrid search setup:

- Dense embeddings: Chroma Cloud Qwen (`Qwen/Qwen3-Embedding-0.6B`)
- Sparse embeddings: Chroma Cloud Splade (`prithivida/Splade_PP_en_v1`)
- Ranking: Reciprocal Rank Fusion (RRF) across dense + sparse search
- Deduplication: `GroupBy(source_document_id)` where available in the SDK

Local fallback is still supported:

```bash
CHROMA_PERSIST_DIR=./chroma_db
```

If Chroma Cloud env vars are unset, the backend falls back to a local persistent client and OpenAI embeddings for session search.

To migrate old local collections into Chroma Cloud:

```bash
python scripts/migrate_local_chroma_to_cloud.py
```

If you want the n8n/Dify external-memory workflow and the backend to share one memory plane, point the workflow at the same Chroma Cloud tenant/database and use the same collection-sharding strategy.

## 3. Tool-calling agent

With `memory_only` and a configured memory webhook, the model can call **`fetch_external_documents`**; with normal routing it calls **`search_web`** (Exa) and **`query_session_memory`**. Implement the webhook to call Dify Knowledge / your DB and return the shapes above.
