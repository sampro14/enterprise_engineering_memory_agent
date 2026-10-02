# Enterprise Engineering Memory Agent

Persistent, temporal, confidence-aware **organizational memory for AI agents**: it tracks *what is true now, what was
true before, and what to distrust*, and learns lessons from repeated failures. Design:
[Enterprise_Engineering_Memory_Agent_Design.md](Enterprise_Engineering_Memory_Agent_Design.md). Evidence and open
questions: [docs/POC_REPORT.md](docs/POC_REPORT.md).

```
client --HTTP--> FastAPI (API key -> tenant) --> MemoryService --> LangGraph flows --> store (Postgres+pgvector | SQLite)
                     |                                 |                                          ^
                     +-- POST /ingest --> jobs table --> worker (claims with SKIP LOCKED) ---------+
providers: create_llm() / create_embedder()  ->  gemini | anthropic | openai | fake   (retries + cache applied once)
```

## Quickstart

### A. Everything in Docker (Postgres + pgvector, API, worker)

```bash
cp .env.example .env            # set GOOGLE_API_KEY (or ANTHROPIC_API_KEY / OPENAI_API_KEY and LLM_PROVIDER)
make up                         # db, migrations, api (http://localhost:8090/docs), worker
make tenant NAME=acme           # prints a tenant id and an API key (shown once)
```

```bash
KEY=mem_...                     # from `make tenant`
curl -s localhost:8090/api/v1/ingest -H "Authorization: Bearer $KEY" -H 'content-type: application/json' -d '{
  "text": "PaymentService uses Stripe as its payment provider.", "uri": "arch.md",
  "source_type": "official_docs", "doc_date": "2025-02-01"}'          # -> 202 {"job_id": "...", "status_url": "..."}
curl -s localhost:8090/api/v1/agent/ask -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"question": "Which payment provider does PaymentService use?"}'
```

`make e2e` runs a full live scenario over HTTP (ingest through the queue, temporal questions, alias resolution, a poisoning
claim landing in review, the review decision, tenant isolation, and the two-phase task flow). Interactive docs: `/docs`.

### B. Local, no infrastructure (SQLite)

```bash
make setup                      # uv sync, creates .env
uv run pytest                   # offline: fake providers, no network
uv run memory-agent demo        # Stripe -> Razorpay story with real LLM calls
uv run memory-agent ingest docs/adr-042.md --source-type architecture_repo --date 2026-09-15
uv run memory-agent ask "Which payment provider does PaymentService use now?" --show-context
uv run memory-agent timeline PaymentService
```

Leave `DATABASE_URL` empty to use SQLite (`MEMORY_DB`); set it to use Postgres.

## LLM providers (factory)

`LLM_PROVIDER` and `EMBED_PROVIDER` are independent, so Claude can generate while Gemini or OpenAI embeds.

| Provider | LLM | Embeddings | Status |
|---|---|---|---|
| `gemini` | yes | yes | **live-tested** (default) |
| `anthropic` | yes | no (Anthropic has no embeddings API) | contract-tested with mocked SDK; not live-tested |
| `openai` | yes | yes | contract-tested with mocked SDK; not live-tested |
| `fake` | scripted | hashed bag-of-words | offline tests, CI |

Add one by writing an adapter and `@register_llm("name")` in [llm/factory.py](src/memory_agent/llm/factory.py). Retries and the
on-disk response cache (`LLM_CACHE`) are applied by the factory, not by adapters. `EMBED_DIM` (default 3072) must match the
vector column; changing it needs a new migration.

## API (design section 26)

All `/api/v1` routes need `Authorization: Bearer <key>`; the tenant comes from the key, never from the body.

| Area | Endpoints |
|---|---|
| Ingest (async) | `POST /ingest` -> 202 + job, `GET /jobs/{id}`, `GET /jobs` |
| Memories | `POST /memories` (sync), `POST /memories/search`, `GET /memories/{id}/history`, `POST /memories/{id}/verify`, `POST /memories/{id}/invalidate`, `DELETE /memories/{id}` |
| Entities | `GET /entities?name=`, `GET /entities/{id}`, `GET /entities/{id}/graph?as_of=`, `POST /entities/{id}/merge`, `POST /entities/merges/{id}/revert` |
| Review | `GET /conflicts`, `GET /reviews`, `POST /reviews/{id}/decision` |
| Agent | `POST /agent/ask`, `POST /agent/tasks`, `GET /agent/tasks/{id}`, `POST /agent/tasks/{id}/outcome`, `GET /tasks/{id}/memory-context` |
| System | `GET /healthz`, `GET /readyz` (database, migrations, provider configuration) |

Errors are `{"error", "detail", "request_id"}` with 401/404/409/422 and 502/503 for provider failures; every response carries
`X-Request-ID`. Logs are JSON on stdout.

Deviations from the design's API: ingestion is one endpoint (`POST /ingest`) instead of `sources` + `run`; tasks are two-phase
(plan, then report the outcome) because execution happens outside the memory service.

## Layout

| Path | What |
|---|---|
| `src/memory_agent/api/` | FastAPI app, auth, schemas, routers |
| `src/memory_agent/services/memory_service.py` | every operation, tenant-scoped; used by CLI, API and worker |
| `src/memory_agent/graph.py` | the LangGraph workflow: ask / plan / execute / outcome (design 27) |
| `src/memory_agent/store/` | store interface, SQLite and Postgres backends, portable schema, migrations |
| `src/memory_agent/llm/` | provider factory, adapters, retry/cache decorators |
| `src/memory_agent/{ingest,extraction,entities,validation,consolidation,retrieval,temporal,confidence}.py` | memory pipeline (design 9, 14-17) |
| `src/memory_agent/{episodes,experience,task_agent}.py` | reflection and experience learning (design 19) |
| `src/memory_agent/{jobs,worker,tenants}.py` | job queue, worker, API keys |
| `eval/`, `docs/POC_REPORT.md` | benchmarks and results |

## Development

```bash
make test               # offline unit tests (SQLite + fake providers)
make test-integration   # same domain, API and queue tests on Postgres (docker compose up -d db)
make lint
make db-shell           # psql into the compose database
```

Every store, domain and API test runs on **both** SQLite and Postgres (the Postgres variants are marked `integration`),
plus a schema-parity test. CI (`.github/workflows/ci.yml`) runs lint, the offline suite, and the Postgres suite against a
pgvector service container.

Operational notes: writes are atomic per document (a failed job leaves nothing half-written); concurrent writers of one
tenant are serialized with a Postgres advisory lock; jobs are retried with exponential backoff, and a job whose worker
died is reclaimed after `JOB_VISIBILITY_TIMEOUT_SECONDS`. Containers run as a non-root user.

## Evaluation

```bash
# temporal facts, contradictions, poisoning, aliases, out-of-order ingestion (seed 0 = development; use 1+ for held-out)
uv run python -m eval.run_eval --seeds 3 --first-seed 1 --systems A,B,BF,C,F
# repeated mistakes on a deployment simulator with hidden organization-specific rules
uv run python -m eval.experience_eval --seeds 3 --first-seed 1 --tasks 24 --systems A,B,C,F
```

Systems: **A** no memory, **B** conversation history, **BF** all documents in context, **C** vector RAG, **F** the full memory
system. Same LLM, prompt and answer schema for all; programmatic grading. See [docs/POC_REPORT.md](docs/POC_REPORT.md) for
results and, importantly, what they do not show (small worlds, synthetic data, experience memory lost to simple baselines).

## Known limitations

- No row-level security in Postgres (tenant scoping is enforced in the store layer and tested, not by the database).
- Deletion propagation is basic: a deleted memory's evidence text and vectors are purged, but derived experiences are not
  re-verified. There are no ACLs on individual memories yet.
- Dates are day-granular. One LLM provider is live-tested. Review approval supports conflicts and agent-inference candidates.
- Ingestion takes text; git/CI/incident connectors are not built.
