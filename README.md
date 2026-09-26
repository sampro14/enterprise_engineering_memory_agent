# Enterprise Engineering Memory Agent — PoC

Proof of concept for the design in [Enterprise_Engineering_Memory_Agent_Design.md](Enterprise_Engineering_Memory_Agent_Design.md):
a persistent memory layer for AI agents that tracks **what is true now, what was true before, and what to distrust**, and
learns lessons from repeated failures.

## Quickstart

```bash
uv sync
cp .env.example .env            # set GOOGLE_API_KEY (Gemini). Model IDs are configurable.
uv run pytest                   # offline: uses a fake LLM/embedder, no network
uv run memory-agent demo        # Stripe -> Razorpay story with real Gemini calls
```

```bash
uv run memory-agent ingest docs/adr-042.md --source-type architecture_repo --date 2026-09-15
uv run memory-agent ask "Which payment provider does PaymentService use now?" --show-context
uv run memory-agent timeline PaymentService
uv run memory-agent reviews                       # conflicts / uncertain candidates awaiting a human
uv run memory-agent decide rev_xxx --approve
```

Source types (authority high → low): `production_config`, `architecture_repo`, `official_docs`, `incident_report`,
`developer`, `agent_inference`.

## How it maps to the design

| Design section | Code |
|---|---|
| 9.4 Extraction (LLM, untrusted text) | [extraction.py](src/memory_agent/extraction.py) |
| 14A Ontology + cardinality | [ontology.py](src/memory_agent/ontology.py) |
| 14B Entity resolution | [entities.py](src/memory_agent/entities.py) |
| 9.5 / 14C Validation, conflict rules, confidence | [validation.py](src/memory_agent/validation.py), [confidence.py](src/memory_agent/confidence.py) |
| 9.6 Consolidation | [consolidation.py](src/memory_agent/consolidation.py) |
| 15 Bitemporal storage and queries | [store/sqlite.py](src/memory_agent/store/sqlite.py), [temporal.py](src/memory_agent/temporal.py) |
| 16-17 Hybrid retrieval | [retrieval.py](src/memory_agent/retrieval.py) |
| 27 LangGraph workflows | [graph.py](src/memory_agent/graph.py) (query flow and task flow) |
| 19 Episodes, reflection, experience promotion | [episodes.py](src/memory_agent/episodes.py), [experience.py](src/memory_agent/experience.py) |
| Human review queue | [review.py](src/memory_agent/review.py) |

Write path: `ingest_text` → extract → resolve entities → validate → consolidate ([ingest.py](src/memory_agent/ingest.py)).

## Evaluation

```bash
# Temporal facts, contradictions, poisoning, aliases, out-of-order ingestion (seed 0 = development; use 1+ for held-out)
uv run python -m eval.run_eval --seeds 3 --first-seed 1 --systems A,B,BF,C,F
# Repeated mistakes on a deployment simulator with hidden organization-specific rules
uv run python -m eval.experience_eval --seeds 3 --first-seed 1 --tasks 24 --systems A,B,C,F
```

Systems: **A** no memory · **B** conversation history (last 10 docs / 8 episodes) · **BF** all documents in context ·
**C** vector RAG (documents, or raw episodes) · **F** the full memory system. All use the same LLM, prompt and answer
schema. Grading is programmatic. Reports are written to `eval/results/`. See [docs/POC_REPORT.md](docs/POC_REPORT.md).

## Deviations from the design doc (PoC simplifications)

- SQLite + brute-force numpy vectors instead of PostgreSQL/pgvector and Neo4j. Graph traversal is SQL over memory rows;
  the `relations` table is a view over memories.
- Dates are day-granular (`YYYY-MM-DD`) for both valid time and system time.
- No `episode_lessons` table: `experience_evidence` links lessons to episodes.
- Review approval supports conflicts and agent-inference candidates only; unknown-relation and ambiguous-entity
  candidates can be rejected but not yet approved.
- No auth, ACLs, deletion propagation, or async write path (MVP scope).
