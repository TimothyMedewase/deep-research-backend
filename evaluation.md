# G3 Backend Evaluation

## 1. System Overview

This backend implements a deep research agent that decomposes a complex user query, gathers evidence with tool calls, stores and retrieves working memory under explicit token and cost limits, and then synthesizes a final answer as a streamed response.

The runtime split is:

- **FastAPI backend**: orchestration, tool use, short-term/session memory, synthesis, SSE streaming
- **n8n query-router workflow**: query routing and optional query rewriting
- **n8n external memory workflow**: long-term memory storage and retrieval across sessions
- **Chroma Cloud**: session research memory and retrieval inside the Python runtime
- **Supabase + pgvector**: long-term external memory owned by the n8n workflow

This matches the G3 objective: the agent answers complex queries under memory and budget constraints, while workflow automation is used for routing and memory management.

## 2. Memory Architecture

The current design is a **hybrid memory system** with two distinct layers.

### Short-term / research-session memory

The backend stores active research context in Chroma through `VectorStore`. Search hits, retrieved documents, and compressed summaries are used to build a rolling research context for the current run.

Key properties:

- **Store**: Chroma Cloud when `CHROMA_API_KEY`, `CHROMA_TENANT`, and `CHROMA_DATABASE` are configured; local persistent Chroma otherwise
- **Dense retrieval**: Chroma Cloud Qwen embeddings in cloud mode
- **Sparse retrieval**: Chroma Cloud Splade embeddings in cloud mode
- **Ranking**: Reciprocal Rank Fusion across dense and sparse search in cloud mode
- **Fallback**: OpenAI `text-embedding-3-small` for local mode
- **Compression**: older context is summarized when the working set exceeds the compression threshold
- **Scope**: the in-app vector store is session-scoped for retrieval, even when collections are sharded by `memory_scope`

This layer is optimized for active research quality, not cross-session persistence.

### Long-term / cross-session memory

The external memory workflow in n8n now actively manages long-term memory in **Supabase (PostgreSQL + pgvector)**. The backend sends `action`, `session_id`, `memory_scope`, `query`, `text`, and `metadata` to the webhook.

Key properties:

- **Primary partition key**: `memory_scope`
- **Session tracking**: `session_id` is stored as metadata, not used as the primary partition key
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Storage**: Supabase table with pgvector embeddings
- **Retrieval**: Supabase RPC similarity search filtered by `memory_scope`
- **API contract**:
  - `retrieve_similar` returns `{ "hits": [...] }`
  - `retrieve_documents` returns `{ "results": [...] }`
  - `store` returns any 2xx JSON body

This layer is what enables memory reuse across sessions for the same user, project, or team.

## 3. Constraint Design

The backend enforces explicit session constraints through `ConstraintTracker` and `MemoryManager`.

Current defaults from the active configuration are:

| Constraint | Value | Rationale |
|---|---:|---|
| Max tokens per session | 8,000 | Caps the rolling session context and prevents uncontrolled prompt growth |
| Compression threshold | 6,000 | Starts compression before the hard limit is reached |
| Max cost per session | $0.50 | Allows substantially deeper research while still bounding spend |
| Max sub-questions | 3 | Limits decomposition breadth to reduce latency and cost |
| Max tool rounds per sub-question | 8 | Prevents open-ended tool loops |

Behavior under constraints:

- If context approaches the token ceiling, older chunks are compressed with `gpt-4o-mini`
- If the token budget is still exceeded after compression, additional ingest is skipped
- If the cost limit is exceeded during sub-question research, the agent stops further research and synthesizes from the context already collected
- Constraint updates from the query-router webhook are now **opt-in** and only apply when `QUERY_ROUTER_APPLY_CONSTRAINTS=true`

This is a soft-fail design: the system prefers to answer from partial context rather than continue spending indefinitely.

## 4. Tool Use and Model Roles

The system uses multiple model roles rather than a single monolithic prompt.

### Decomposition

The backend first decomposes the original query into sub-questions with `QueryDecomposer`.

### Tool-calling research loop

Each sub-question is handled by a tool-calling loop that can use:

- `query_session_memory`
- `search_web`
- `fetch_external_documents` when external memory is configured and the route is `memory_only`

This loop is handled in Python and streams intermediate events such as:

- `route`
- `thinking`
- `decompose`
- `tool_call`
- `search`
- `memory_hit`
- `compress`
- `token_update`
- `answer`
- `done`

### Final synthesis

After evidence gathering, the backend performs a final synthesis pass and streams the answer incrementally as SSE `answer` events.

This design keeps tool execution deterministic in the server while using the LLM for decomposition, tool choice, and synthesis.

## 5. Workflow Role (n8n)

### Query router workflow

The query-router webhook receives:

```json
{ "query": "user text", "session_id": "uuid" }
```

and returns:

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

Supported routes:

- `deep_research`
- `memory_only`
- `blocked`

In the current backend, router-supplied constraints are ignored unless `QUERY_ROUTER_APPLY_CONSTRAINTS=true`. This prevents n8n from silently overriding the server’s configured budgets.

### External memory workflow

The external memory webhook actively manages long-term memory in Supabase with pgvector.

The workflow:

1. receives `store`, `retrieve_similar`, or `retrieve_documents`
2. generates embeddings with OpenAI
3. writes to or queries Supabase
4. returns normalized JSON for the backend to merge into the research process

This is no longer a placeholder adapter. It is now a real long-term memory plane partitioned by `memory_scope`.

## 6. Current Strengths

- The system now has a clear separation between **short-term research memory** and **long-term external memory**
- The research loop is bounded by explicit token and cost controls
- The final answer is streamed progressively to the frontend rather than buffered until completion
- Chroma Cloud gives the in-app memory layer strong retrieval with hybrid dense + sparse ranking
- n8n external memory now supports cross-session recall through `memory_scope`
- The backend no longer treats “cost limit reached, synthesizing from available context” as a fatal error event

## 7. Known Limitations

- The short-term Chroma retrieval path is still session-scoped, so user-level long-term recall depends on the external memory webhook rather than the in-app vector store
- Compression is lossy and can still drop low-frequency details or exact phrasing
- The query-router can still rewrite the query, so poor routing or rewriting logic in n8n can reduce answer quality
- The external memory workflow depends on OpenAI embeddings and Supabase RPC correctness; operational failures there degrade long-term recall
- The synthesis step currently streams free-form answer text rather than a structured citation object model

## 8. Assessment Against the G3 Objective

This implementation satisfies the core G3 requirements:

- **Deep research agent**: yes
- **Memory/token constraints**: yes
- **LLM with tool use**: yes
- **Workflow integration for routing + memory**: yes

The strongest architectural point is the split between:

- **FastAPI + Chroma** for active research-time memory under tight budgets
- **n8n + Supabase/pgvector** for persistent long-term memory across sessions

That gives the agent both immediate working memory and a durable external memory plane, while keeping orchestration and answer synthesis in the Python runtime.
