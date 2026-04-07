# n8n / Dify integration

The FastAPI service remains the runtime for research orchestration, tool execution, session memory, and final synthesis. **n8n** or **Dify** integrate through HTTPS webhooks for query routing and long-term memory.

## Runtime split

- **FastAPI backend**: decomposition, tool use, session-scoped memory, SSE answer streaming
- **Chroma Cloud**: short-term/session research memory inside the backend
- **n8n query router**: route selection and optional query rewriting
- **n8n external memory workflow**: long-term memory storage and retrieval
- **Supabase + pgvector**: durable external memory store used by the n8n workflow

**By default this integration is required:** `N8N_INTEGRATION_REQUIRED=true` makes the app fail at startup unless `QUERY_ROUTER_WEBHOOK_URL` and `EXTERNAL_MEMORY_WEBHOOK_URL` are configured. For local development without workflows, set `N8N_INTEGRATION_REQUIRED=false`.

## 1. Query router (`QUERY_ROUTER_WEBHOOK_URL`)

**POST** JSON body:

```json
{ "query": "user text", "session_id": "uuid" }
```

Expected response:

```json
{
  "route": "deep_research",
  "query": "optional rewritten query",
  "constraints": {
    "max_sub_questions": 3,
    "max_cost_usd": 0.5,
    "max_tokens": 8000
  }
}
```

- `route`: `deep_research` | `memory_only` | `blocked`
- `query`: optional rewritten query
- `constraints`: optional budget patch from the workflow

Important:

- Router-supplied constraints are only applied when `QUERY_ROUTER_APPLY_CONSTRAINTS=true`
- If `QUERY_ROUTER_APPLY_CONSTRAINTS=false`, the backend keeps its own configured limits and ignores the router `constraints` block
- Optional `QUERY_ROUTER_WEBHOOK_BEARER` sends `Authorization: Bearer …`

## 2. External memory (`EXTERNAL_MEMORY_WEBHOOK_URL`)

One webhook URL can handle three actions. Branch on `action` in n8n with a Switch node.

### Incoming request contract

| `action` | Body fields |
|----------|-------------|
| `retrieve_similar` | `query`, `session_id`, `memory_scope` |
| `retrieve_documents` | `query`, `session_id`, `memory_scope` |
| `store` | `session_id`, `memory_scope`, `text`, `metadata` |

### Expected response contract

| `action` | Response |
|----------|----------|
| `retrieve_similar` | `{ "hits": [ { "text", "metadata", "distance"? } ] }` |
| `retrieve_documents` | `{ "results": [ { "url", "title", "content" } ] }` |
| `store` | any JSON with 2xx status |

Optional:

- `EXTERNAL_MEMORY_STORE_WEBHOOK_URL` overrides the URL for `store` only
- `EXTERNAL_MEMORY_WEBHOOK_BEARER` adds `Authorization: Bearer …`

## 3. Long-term memory design

The external memory workflow now actively manages long-term memory in **Supabase (PostgreSQL + pgvector)**.

### Partitioning

- **Primary partition key**: `memory_scope`
- Examples: `user:john`, `project:alpha`, `team:research`
- `session_id` is stored as metadata for traceability and auditing
- Cross-session recall works by querying within the same `memory_scope`

### Embeddings

- Provider: **OpenAI**
- Model: `text-embedding-3-small`

### Store flow

For `action = "store"`:

1. n8n receives `session_id`, `memory_scope`, `text`, `metadata`
2. n8n generates an embedding with OpenAI
3. n8n inserts the record into Supabase
4. n8n returns a 2xx JSON response such as:

```json
{ "stored": true }
```

### Similar retrieval flow

For `action = "retrieve_similar"`:

1. n8n receives `query`, `session_id`, `memory_scope`
2. n8n generates a query embedding with OpenAI
3. n8n calls a Supabase RPC similarity function such as `match_memories`
4. n8n returns:

```json
{
  "hits": [
    {
      "text": "matched memory text",
      "metadata": {
        "session_id": "abc-123",
        "topic": "AI"
      },
      "distance": 0.12
    }
  ]
}
```

If nothing matches:

```json
{ "hits": [] }
```

### Document retrieval flow

For `action = "retrieve_documents"`:

1. n8n performs the same scoped vector retrieval
2. n8n formats results as document-style records
3. n8n returns:

```json
{
  "results": [
    {
      "url": "",
      "title": "Memory from session abc-123",
      "content": "matched memory text"
    }
  ]
}
```

If nothing matches:

```json
{ "results": [] }
```

## 4. Suggested Supabase schema

```sql
create extension if not exists vector;

create table memories (
  id uuid primary key default gen_random_uuid(),
  memory_scope text not null,
  session_id text not null,
  text text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1536),
  created_at timestamptz default now()
);

create index on memories using ivfflat (embedding vector_cosine_ops);
create index idx_memories_scope on memories(memory_scope);
create index idx_memories_session on memories(session_id);
```

Example RPC shape:

```sql
create function match_memories(
  query_embedding vector(1536),
  match_memory_scope text,
  match_threshold float default 0.8,
  match_count int default 5
)
returns table (
  id uuid,
  session_id text,
  text text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select
    m.id,
    m.session_id,
    m.text,
    m.metadata,
    1 - (m.embedding <=> query_embedding) as similarity
  from memories m
  where m.memory_scope = match_memory_scope
    and 1 - (m.embedding <=> query_embedding) >= match_threshold
  order by m.embedding <=> query_embedding
  limit match_count;
$$;
```

## 5. Backend session memory (Chroma Cloud)

The backend uses **Chroma Cloud** for active research-session memory when these env vars are set:

```bash
CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=...
CHROMA_TENANT=25b8f9e7-2c7b-4dc5-8202-d077e96396db
CHROMA_DATABASE=research-agent
CHROMA_COLLECTION_PREFIX=research-memory
CHROMA_DEFAULT_SCOPE=default
```

Behavior:

- The backend uses `chromadb.CloudClient(...)`
- Collections are named from `CHROMA_COLLECTION_PREFIX` plus a normalized scope
- Pass `memory_scope` in `POST /research` to shard memory at the collection level
- Retrieval inside the Python runtime still filters by `session_id`, so this remains session-scoped working memory
- Documents are chunked before insertion to stay under Chroma record limits
- Chroma Cloud retrieval uses hybrid dense + sparse search with RRF

Local fallback is still supported with:

```bash
CHROMA_PERSIST_DIR=./chroma_db
```

If Chroma Cloud env vars are unset, the backend falls back to local persistent Chroma and OpenAI embeddings for session retrieval.

## 6. Tool-calling behavior

The research loop can use:

- `query_session_memory`
- `search_web`
- `fetch_external_documents` when the route is `memory_only` and the external memory workflow is configured

That means:

- **Session memory** comes from the in-app Chroma layer
- **Long-term memory** comes from the external n8n workflow
- Both can contribute context to the same research run

## 7. Recommended environment variables

```bash
N8N_INTEGRATION_REQUIRED=true

QUERY_ROUTER_WEBHOOK_URL=https://your-n8n-instance.com/webhook/query-classify
QUERY_ROUTER_APPLY_CONSTRAINTS=false

EXTERNAL_MEMORY_WEBHOOK_URL=https://your-n8n-instance.com/webhook/memory

CHROMA_HOST=api.trychroma.com
CHROMA_API_KEY=...
CHROMA_TENANT=25b8f9e7-2c7b-4dc5-8202-d077e96396db
CHROMA_DATABASE=research-agent
CHROMA_COLLECTION_PREFIX=research-memory
CHROMA_DEFAULT_SCOPE=default
```

## 8. Practical guidance

- Use **Chroma Cloud** for short-term research memory inside the backend
- Use **Supabase + pgvector** for long-term external memory in n8n
- Use `memory_scope` for cross-session isolation and sharing
- Keep router constraints disabled by default unless you explicitly want n8n to control budgets
- Do not use localhost-backed services from n8n Cloud
