# Enterprise Engineering Memory Agent — System Design

## 1. Executive Summary

The **Enterprise Engineering Memory Agent** is a persistent-memory multi-agent system designed to help AI agents accumulate, validate, organize, retrieve, and learn from knowledge acquired across long-running enterprise workflows.

Traditional RAG primarily answers:

> Which documents are relevant to this question?

The Enterprise Memory Agent addresses:

> What has the organization learned over time, what is currently true, what changed, what happened before, and which previous experiences should influence the current decision?

The system combines:

- Short-term conversational memory
- Episodic memory
- Semantic memory
- Procedural memory
- Experience memory
- Vector retrieval
- Temporal knowledge graphs
- Confidence-aware memory
- Contradiction detection
- Memory consolidation
- Memory decay and invalidation
- Multi-agent orchestration
- Reflection and experience extraction

The initial target domain is **enterprise software engineering**, where the agent can remember services, APIs, databases, architecture decisions, incidents, deployments, teams, bugs, solutions, and historical changes.

---

# 2. Problem Statement

Current AI agents commonly have these limitations:

1. Useful information is lost between sessions.
2. Conversation history becomes too large to manage efficiently.
3. Vector memory retrieves semantic similarity but does not naturally model relationships or temporal state.
4. Old information may remain active after becoming obsolete.
5. Contradictory memories can be retrieved simultaneously.
6. Agents can repeatedly make mistakes they have already encountered.
7. Raw conversation history is not equivalent to structured knowledge.
8. Agents have limited mechanisms for learning generalized experiences from repeated tasks.

Example:

```text
January:
Payment service uses Stripe.

May:
Team plans migration to Razorpay.

September:
Migration completed.

October:
"Which payment provider do we currently use?"
```

A reliable memory system should answer:

```text
Current:
Razorpay

Historical:
Stripe

Migration:
Completed in September

Evidence:
Deployment configuration + architecture document + migration record
```

---

# 3. Vision

Build an AI agent that can operate over long periods while maintaining a continuously updated representation of enterprise knowledge.

```text
Observe
   ↓
Learn
   ↓
Validate
   ↓
Remember
   ↓
Consolidate
   ↓
Reason
   ↓
Act
   ↓
Observe outcome
   ↓
Learn again
```

---

# 4. Goals

## Primary Goals

1. Maintain persistent memory across sessions.
2. Represent knowledge using semantic vectors and structured relationships.
3. Track historical and current facts.
4. Detect contradictory information.
5. Resolve conflicts using source authority, timestamps, verification, and confidence.
6. Distinguish temporary events from durable knowledge.
7. Extract reusable experiences from previous tasks.
8. Retrieve relevant memories during future tasks.
9. Prevent repeated mistakes using experience memory.
10. Evaluate memory quality quantitatively.
11. Support multi-agent workflows.
12. Provide an inspectable memory UI.

## Secondary Goals

- Support multiple LLM providers.
- Support configurable vector databases.
- Support configurable graph databases.
- Support human approval for selected memory updates.
- Provide memory export/import.
- Provide benchmark datasets for research experiments.

---

# 5. Non-Goals

The initial system will not attempt to:

- Build AGI.
- Store every conversation indefinitely.
- Treat every generated statement as factual.
- Automatically overwrite authoritative enterprise data without verification.
- Replace enterprise source-of-truth systems.
- Make irreversible production changes solely from memory.

Memory is a **reasoning aid**, not the authoritative source for high-impact operational facts.

---

# 6. Initial Product: Enterprise Engineering Memory Agent

The initial implementation focuses on software engineering organizations.

The system remembers:

```text
Services
APIs
Databases
Repositories
Teams
Owners
Architecture decisions
Deployments
Incidents
Bugs
Solutions
Technical decisions
Configuration changes
Past failures
Successful fixes
Operational procedures
```

Example questions:

- Why does the payment service use Oracle?
- What database did this service use before Oracle?
- Have we seen this deployment failure before?
- What fixed the previous incident?
- Who owns the payment service?
- What changed in the payment architecture last month?
- Which previous deployment had a similar failure?

---

# 7. Core Design Principle

## Do not treat memory as a vector database.

Use multiple complementary representations.

```text
                  MEMORY SYSTEM
                       |
       +---------------+---------------+
       |               |               |
       v               v               v
 Vector Store      Graph Store    Episodic Store
       |               |               |
 Semantic          Relations       Events /
 Similarity        + Time          Experiences
       |               |               |
       +---------------+---------------+
                       |
                       v
                Memory Retriever
```

### Vector Store

Answers:

> What memories are semantically similar?

### Knowledge Graph

Answers:

> How are these entities related?

### Episodic Store

Answers:

> What happened during a previous task?

### Relational Store

Answers:

> What exactly happened, when, and with what metadata?

In the MVP the relational store is PostgreSQL and acts as the **system of record**. The episodic store and the vector index (pgvector) live in the same database, and the graph store can start as relation tables in PostgreSQL (see Section 24).

---

# 8. High-Level Architecture

```text
                              USER
                                |
                                v
                       +----------------+
                       |  Orchestrator  |
                       +-------+--------+
                               |
                               v
                         +-----------+
                         |   Task    |
                         |   Agent   |
                         +-----+-----+
                               |
                     Need historical context?
                               |
                               v
                    +----------------------+
                    | Memory Retrieval     |
                    | Agent                |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
        +-----------+    +-----------+    +-----------+
        | Vector DB |    | Graph DB  |    | Episodic  |
        |           |    |           |    | Store     |
        +-----+-----+    +-----+-----+    +-----+-----+
              |                |                |
              +----------------+----------------+
                               |
                               v
                       Context Builder
                               |
                               v
                         Task Reasoning
                               |
                               v
                             Action
                               |
                               v
                            Outcome
                               |
                               v
                     Memory Extraction Agent
                               |
                               v
                     Memory Validation Agent
                               |
                  +------------+------------+
                  |                         |
                REJECT                    ACCEPT
                  |                         |
                  |                         v
                  |                 Memory Consolidator
                  |                         |
                  |                         v
                  |                  Graph Update Agent
                  |                         |
                  +-------------------------+
```

---

# 9. Multi-Agent Architecture

## 9.1 Orchestrator Agent

Coordinates the workflow:

- Interpret task
- Decide whether memory is needed
- Route to appropriate agents
- Track workflow state
- Coordinate retrieval and reasoning
- Trigger memory write after task completion

## 9.2 Task Agent

Performs the actual enterprise task:

- Investigate incident
- Analyze logs
- Modify code
- Explain architecture
- Investigate API failure
- Generate SQL
- Analyze deployment

The Task Agent does not own long-term memory; it requests memory through the Memory Retrieval Agent.

## 9.3 Memory Retrieval Agent

Decides:

- Whether memory is required
- Which memory types are relevant
- Which entities should be retrieved
- Whether vector search is sufficient
- Whether graph traversal is required
- Whether historical events are required

## 9.4 Memory Extraction Agent

Extracts candidate memories from interactions.

Example:

```text
"We migrated the payment service from Stripe to Razorpay
because enterprise customers required a specific settlement workflow."
```

Possible output:

```json
{
  "memories": [
    {
      "type": "technical_fact",
      "subject": "PaymentService",
      "relation": "uses",
      "object": "Razorpay"
    },
    {
      "type": "historical_fact",
      "subject": "PaymentService",
      "relation": "previously_used",
      "object": "Stripe"
    },
    {
      "type": "decision",
      "subject": "PaymentService",
      "relation": "migration_reason",
      "object": "enterprise settlement requirements"
    }
  ]
}
```

## 9.5 Memory Validation Agent

No candidate memory should automatically become trusted long-term memory.

Evaluate:

- factuality
- usefulness
- duplication
- contradiction
- temporariness
- evidence support
- source authority
- sensitive information

Possible decisions:

```text
ACCEPT
REJECT
UPDATE_EXISTING
MARK_CONFLICT
REQUIRE_HUMAN_REVIEW
```

## 9.6 Memory Consolidator

Converts fragmented memories into coherent knowledge.

```text
Memory 1:
Payments use Stripe.

Memory 2:
Payments are migrating to Razorpay.

Memory 3:
Migration completed.

Memory 4:
Razorpay is now production provider.
```

Consolidated representation:

```text
PaymentService

Stripe
  used_from: 2025-01
  used_until: 2026-09-10

Razorpay
  used_from: 2026-09-11
  current: true
```

## 9.7 Knowledge Graph Agent

Manages entities and relationships:

```text
PaymentService
   |
   +-- uses ----------------> Razorpay
   |
   +-- previously_used -----> Stripe
   |
   +-- owned_by ------------> PaymentsTeam
   |
   +-- depends_on ----------> Oracle
   |
   +-- has_incident --------> Incident-421
```

## 9.8 Planner Agent

Decomposes a task into steps, decides which steps need memory, and revises the plan when retrieved memory contradicts the initial assumptions. Its output is `current_plan` in the LangGraph state (Section 27).

## 9.9 Reflection Agent

Runs after task completion and periodically in the background. It compares outcomes with retrieved memory, flags memories that were used but proved wrong (feeding conflict handling in Section 14C), and proposes lessons for Experience Memory (Section 19).

## 9.10 Ingestion Agent

Extraction from agent interactions alone (9.4) is not enough for enterprise use. The Ingestion Agent pulls candidate memories from sources of truth through connectors:

- Git repositories, ADRs, README and architecture documents
- CI/CD and deployment records
- Service configuration (e.g., `deployment.yaml`)
- Incident and ticketing systems
- Service catalog and ownership metadata

Every ingested candidate carries its source, URI and authority level (Section 21) and passes through the same Memory Validation Agent as any other candidate. Connector output is treated as untrusted text (see Section 23).

---

# 10. Memory Types

## Short-Term Memory

Current task state:

```text
Current conversation
Current plan
Current observations
Current tool results
```

Lifetime: task/session.

## Episodic Memory

Represents events:

```text
Deployment #928
Incident #421
Customer interaction
Previous debugging session
Previous migration
```

## Semantic Memory

Generalized facts:

```text
PaymentService uses Razorpay.
PaymentsTeam owns PaymentService.
Oracle is the primary database.
```

## Procedural Memory

How to perform tasks:

```text
Deployment Procedure:

1. Run tests
2. Run migration
3. Build image
4. Deploy staging
5. Run smoke tests
6. Promote
```

## Experience Memory

Lessons learned:

```text
Experience:

Deployments involving database schema changes
should execute migrations before application deployment.

Evidence:
3 historical deployment failures.

Confidence:
0.94 (illustrative)
```

---

# 11. Memory Lifecycle

```text
                  EXPERIENCE
                       |
                       v
                Memory Extraction
                       |
                       v
                Importance Scoring
                       |
                       v
                  Deduplication
                       |
                       v
              Contradiction Detection
                       |
                       v
                 Source Validation
                       |
                       v
                 Memory Storage
                       |
                       v
                Consolidation
                       |
                       v
              Knowledge Graph Update
                       |
                       v
                  Retrieval
                       |
                       v
                   Reasoning
                       |
                       v
                    Action
                       |
                       v
                   Outcome
                       |
                       +--------> EXPERIENCE
```

## Synchronous vs Asynchronous Execution

A single task can trigger several LLM calls on the write path (extraction, validation, consolidation, graph update). To keep task latency predictable:

| Step | Mode | Reason |
|---|---|---|
| Retrieval, re-ranking, conflict check | Synchronous | Needed before reasoning |
| Candidate extraction | Asynchronous (after the task returns) | Not on the user's critical path |
| Validation | Asynchronous; synchronous for high-impact memories | Cheap rules first, LLM only when needed |
| Consolidation, decay, experience mining | Background jobs | Batch-friendly and periodic |
| Graph updates | Serialized per entity (locking or an ordered queue) | Avoid concurrent writers producing conflicting `valid_until` values |

Writes must be idempotent (keyed by source + content hash) so retries do not create duplicate memories.

---

# 12. Memory Importance Scoring

A memory should not be retained simply because it appeared in a conversation.

Candidate scoring can consider:

```text
Importance =
    Relevance
  + Reusability
  + Confidence
  + Recurrence
  + Business Value
  - Temporariness
```

Example:

```text
"I am testing the service today."
→ Low importance

"Payment service migrated to Razorpay."
→ High importance
```

Weights should be configurable and evaluated experimentally.

---

# 13. Confidence-Aware Memory

Each memory should have metadata:

```json
{
  "memory_id": "mem_8291",
  "subject": "PaymentService",
  "relation": "uses",
  "object": "Razorpay",
  "confidence": 0.96,
  "source_count": 4,
  "source_authority": "high",
  "last_verified": "2026-09-20",
  "created_at": "2026-09-11",
  "last_accessed": "2026-09-25"
}
```

Confidence can depend on:

- Source authority
- Number of independent sources
- Recency
- Verification status
- Consistency
- Explicit user confirmation
- Configuration evidence

---

# 14. Contradiction Detection

Example:

```text
Memory A:
Database = PostgreSQL

Memory B:
Database = Oracle
```

Evaluate:

```text
Timestamp
Source authority
Verification
Context
Confidence
Current configuration
```

Possible result:

```text
Current:
Oracle

Historical:
PostgreSQL
```

The older fact remains available as historical knowledge unless explicitly invalidated or removed under retention policy.

---

# 14A. Ontology and Relation Cardinality

Contradiction detection is only meaningful if the system knows which relations can have one value and which can have many. Maintain a versioned relation schema:

| Relation | Cardinality | Contradiction rule |
|---|---|---|
| `primary_database` | single-valued at a point in time | Different objects with overlapping validity → conflict |
| `owned_by` | single-valued | Same as above |
| `uses` (payment provider) | single-valued per role | Conflict per role |
| `depends_on` | multi-valued | No conflict; add another edge |
| `has_incident` | multi-valued | No conflict |

Rules:

- Every relation type declares its cardinality, allowed subject/object entity types, and whether it is temporal.
- Unknown relations extracted by the LLM are stored as `proposed` and reviewed before being added to the schema.
- "PostgreSQL vs Oracle" is a conflict only for a single-valued relation such as `primary_database`. For `depends_on` it is two valid facts.

---

# 14B. Entity Resolution

`PaymentService`, `payments-svc` and `payment-api` may all refer to one entity. Without resolution the graph fragments, and contradictions go undetected because each alias looks like a different service.

Pipeline for each extracted mention:

1. Normalize the name (case, separators, known prefixes and suffixes).
2. Exact match against entity names and stored aliases.
3. Fuzzy and embedding match against entities of the same type within the tenant.
4. Disambiguate using context (repository, team, environment).
5. Decide: `MATCH` (above a high threshold), `NEW_ENTITY`, or `REVIEW` (ambiguous).

Rules:

- Aliases are stored in an `entity_aliases` table (Section 25) with source and confidence.
- Merges are reversible: keep a merge log so a wrong merge can be split.
- Never auto-merge across tenants, and never auto-merge entities of different types.
- Resolution accuracy is an evaluation metric (Section 31).
- **PoC finding:** compare the distinctive part of a name (drop generic tokens such as "service", "svc", "team") and require string similarity of at least 0.75 even for embedding matches. Embeddings alone confuse similar-looking names (`InventoryService` vs `InvoiceService`), which flooded the review queue. Measured: distinct pairs 0.50-0.62, true variants 0.82+.

---

# 14C. Conflict Resolution and Confidence Calculation

When a candidate memory conflicts with an existing one on a single-valued relation, apply these rules in order:

1. **Scope check**: are they about the same entity, environment and time interval? If not, there is no conflict.
2. **Verified current source of truth** (e.g., production configuration) wins over any memory-derived claim.
3. **Explicit, newer verified evidence** wins over older evidence, even when the older evidence has higher source authority (consistent with Section 21).
4. Otherwise compare the confidence scores below.
5. If the score difference is under a configurable margin, mark `MARK_CONFLICT` and route to human review. Do not silently pick one.

The losing memory is not deleted. Its `valid_until` is set and it is linked through `supersedes`, so it remains available for historical queries.

Confidence is a configurable weighted combination, for example:

```text
confidence = clamp(
    w_a * authority_score
  + w_s * independent_source_score     # diminishing returns, e.g. 1 - 0.5^n
  + w_v * verification_score           # unverified / verified / user-confirmed
  + w_r * recency_score                # decays with time since last_verified
  - w_c * contradiction_penalty
)
```

Weights and functional forms are hyperparameters to be tuned on the benchmark (Section 36). No particular values are claimed here.

---

# 15. Temporal Memory

Important relationships can contain:

```text
valid_from
valid_until
observed_at
created_at
last_verified
```

Example:

```json
{
  "subject": "PaymentService",
  "relation": "uses",
  "object": "Razorpay",
  "valid_from": "2026-09-11",
  "valid_until": null,
  "confidence": 0.96
}
```

This supports:

- What does the system use now?
- What did the system use before the migration?
- When did the migration happen?

## Bitemporal semantics

The model tracks two time dimensions:

- **Valid time** (`valid_from`, `valid_until`): when the fact was true in the real world.
- **System time** (`created_at`, `observed_at`, and `recorded_until` once a record is superseded): when the memory system learned about the fact or stopped believing it.

Query semantics:

- `current`: `valid_until IS NULL` (or in the future) and the record is not superseded.
- `as of T` (valid time): `valid_from <= T < COALESCE(valid_until, infinity)`.
- `as known on D` (system time): what the system believed on date D, ignoring later corrections.
- An unknown `valid_until` means "still true as far as we know". It is closed when a superseding fact is accepted (`valid_until` = the new fact's `valid_from`).
- **Backdated corrections** (learning in October that the migration actually happened in September) insert a record with the earlier `valid_from` and a new `observed_at`. The previous record is retained, so "as known on" queries stay reproducible.

---

# 16. Memory Retrieval Pipeline

```text
User Task
   |
   v
Intent Analysis
   |
   v
Entity Extraction
   |
   v
Memory Need Detection
   |
   +---- No memory needed ---> Continue
   |
   +---- Memory needed
              |
              v
       Hybrid Retrieval
              |
      +-------+-------+
      |       |       |
      v       v       v
   Vector   Graph   Episodic
      |       |       |
      +-------+-------+
              |
              v
       Re-ranking
              |
              v
       Conflict Check
              |
              v
       Context Builder
              |
              v
        Task Agent
```

---

# 17. Hybrid Retrieval

The system combines:

### Semantic Retrieval
Find similar memories.

### Entity Retrieval
Find memories connected to entities.

### Graph Traversal
Follow relationships.

### Temporal Filtering
Prefer memories valid for the requested time.

### Source Filtering
Prefer authoritative evidence.

### Re-ranking

Rank by:

```text
Relevance
Confidence
Freshness
Source Authority
Temporal Fit
Relationship Proximity
```

---

# 18. Example Retrieval

Question:

> Why did the payment API start failing after the latest deployment?

Potential retrieval:

```text
Entity:
PaymentAPI

Related:
PaymentService
Deployment-928
Database
ConnectionPool
Incident-421
```

Graph:

```text
Deployment-928
      |
      +--> PaymentAPI
      |
      +--> Database
      |
      +--> Configuration
```

Historical experience:

```text
Incident-421:
Connection pool misconfiguration caused similar failure.
```

Final context:

```text
Current deployment
+
Current configuration
+
Related historical incident
+
Known solution
```

---

# 19. Experience Learning

Repeated events should be converted into generalized knowledge.

```text
Deployment #12
→ Failure: migration ordering

Deployment #18
→ Failure: migration ordering

Deployment #31
→ Failure: migration ordering
```

Create:

```text
Experience:

When a deployment contains database schema changes,
execute database migrations before application deployment.

Evidence:
3 historical incidents.

Confidence:
0.94 (illustrative)
```

Future deployment:

```text
New deployment
      |
      v
Retrieve experience
      |
      v
"Run migration first"
      |
      v
Avoid previous failure
```

## Promotion rules (episodes → experience)

Repeated episodes become an experience only when all of the following hold (thresholds are configurable):

1. **Clustering**: episodes share a normalized failure/outcome signature (e.g., same error class and same affected entity type), grouped by embedding similarity plus rule-based keys.
2. **Minimum evidence**: at least N independent episodes (proposed default: 3) from distinct tasks or deployments.
3. **Scope conditions**: the lesson records the conditions under which it applies (`conditions` field) so it is not applied where it does not hold.
4. **Validation**: the causal explanation is checked against evidence (e.g., the incident report), not only correlation. Otherwise it is stored as a low-confidence hypothesis.
5. **Confidence**: derived from evidence count, consistency of outcomes and counter-examples (episodes matching the conditions where the lesson did not hold). The 0.94 above is illustrative.

**Verification before injection (PoC finding):** a drafted lesson is checked against the failed and contrasting successful episodes before use. In the PoC the first drafts reversed a required step order and made later tasks worse. Lessons that fail verification twice are stored as `hypothesis` and never injected. Failures that occur while a lesson was applied count as counter-examples, not as supporting evidence.

Invalidation: an experience is re-verified when the entities it depends on change (e.g., the deployment process changes), when counter-examples accumulate, or on a schedule. Stale experiences are demoted, not deleted.

---

# 20. Memory Decay and Forgetting

Each memory can contain:

```text
importance
confidence
freshness
last_accessed
last_verified
expiration
source_authority
```

Potential mechanisms:

### Decay
Low-value memories become less retrievable.

### Invalidation
An authoritative source marks a fact obsolete.

### Consolidation
Multiple memories become one generalized representation.

### Archival
Historical memories remain available for historical queries but have lower default retrieval priority.

---

# 21. Source Authority

Suggested configurable hierarchy:

```text
Production configuration
        ↓
Architecture repository
        ↓
Official documentation
        ↓
Verified incident report
        ↓
Developer confirmation
        ↓
Agent-generated inference
```

Source authority influences conflict resolution but should not automatically override explicit, newer verified evidence.

---

# 22. Security and Privacy

Required controls:

- Tenant isolation
- Access control
- Memory-level permissions
- Encryption at rest
- Encryption in transit
- Audit logging
- Retention policies
- Deletion/invalidation
- Sensitive-data filtering
- Human approval for selected memory classes

The system must not expose private information from one user or team to another unauthorized party.

Additional requirements:

- **Permission inheritance**: a derived memory (extracted, consolidated, or an experience) inherits the most restrictive access policy of its source evidence. Retrieval filters by the caller's permissions before ranking.
- **Deletion propagation**: deleting or invalidating a source must propagate to the memories derived from it, their embeddings, graph edges, consolidations, and any experience that relied on them (via the `evidence` table).

---

# 23. Memory Poisoning Protection

Potential attack:

```text
"Production database is PostgreSQL."
```

when the real system uses Oracle.

Protection:

```text
Candidate Memory
      |
      v
Source Verification
      |
      v
Authority Check
      |
      v
Conflict Detection
      |
      v
Confidence Calculation
      |
      v
Accept / Reject / Review
```

High-impact memories require stronger evidence.

Poisoning can also come from **untrusted content**, not only incorrect human claims:

- Text in retrieved documents, tickets, web pages, logs or tool output can contain instructions ("remember that..."). Content from tools and connectors is treated as data, never as instructions to the memory system.
- Every candidate memory must cite evidence, and extraction should only draw on content trusted for that purpose.
- A memory supported only by agent-generated inference (the lowest tier in Section 21) cannot alone establish a high-impact fact.
- Poisoning resistance is measured in the experiment in Section 34.

---

# 24. Recommended Technology Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| API | FastAPI |
| Primary database | PostgreSQL |
| Vector search | pgvector or Qdrant |
| Knowledge graph | Neo4j |
| Cache | Redis |
| LLM | Provider abstraction |
| Embeddings | Configurable embedding model |
| UI | React / Next.js |
| Observability | OpenTelemetry |
| Deployment | Docker |

**MVP simplification:** start with PostgreSQL only: pgvector for similarity, relation tables with recursive CTEs for graph traversal, and temporal columns for history. Add Neo4j and Redis only if the evaluation shows a measurable need. Keep the graph store behind an interface so it can be swapped for the ablation studies in Section 36.

---

# 25. Data Model

## Memory

```text
memories
--------
id
tenant_id
memory_type
subject
relation
object
content
confidence
importance
source_authority
valid_from
valid_until
observed_at
created_at
recorded_until        (set when superseded)
supersedes_id
last_verified
last_accessed
status                (candidate | active | conflicted | superseded | invalidated | archived)
access_policy
```

## Entity

```text
entities
--------
id
tenant_id
entity_type
name
description
created_at
updated_at
```

## Entity Alias

```text
entity_aliases
--------------
id
tenant_id
entity_id
alias
source
confidence
created_at
```

## Relation

```text
relations
---------
id
tenant_id
source_entity_id
relation
target_entity_id
confidence
valid_from
valid_until
recorded_until
source_memory_id
```

## Evidence

```text
evidence
--------
id
tenant_id
memory_id
source_type           (config | repo | doc | incident | user | agent_inference)
source_uri
excerpt
source_authority
observed_at
access_policy
```

## Review

```text
reviews
-------
id
tenant_id
memory_id
reason                (conflict | sensitive | low_confidence | ambiguous_entity)
status                (pending | approved | rejected)
reviewer
decided_at
```

## Episode

```text
episodes
--------
id
tenant_id
task_id
summary
outcome
timestamp

episode_entities   (episode_id, entity_id)
episode_lessons    (episode_id, experience_id)
```

## Experience

```text
experiences
-----------
id
tenant_id
status
pattern
lesson
evidence_count
confidence
conditions
recommended_action
created_at
last_verified

experience_evidence   (experience_id, episode_id)
```

---

# 26. API Design

```http
# Memories
POST   /api/v1/memories
POST   /api/v1/memories/search
GET    /api/v1/memories/{memory_id}/history
POST   /api/v1/memories/{memory_id}/verify
POST   /api/v1/memories/{memory_id}/invalidate
DELETE /api/v1/memories/{memory_id}            # with deletion propagation (Section 22)

# Entities and graph
GET    /api/v1/entities/{entity_id}
GET    /api/v1/entities/{entity_id}/graph?as_of=YYYY-MM-DD
POST   /api/v1/entities/{entity_id}/merge      # entity resolution, reversible

# Conflicts and review
GET    /api/v1/conflicts
GET    /api/v1/reviews?status=pending
POST   /api/v1/reviews/{review_id}/decision

# Ingestion
POST   /api/v1/ingest/sources
POST   /api/v1/ingest/{source_id}/run

# Agent
POST   /api/v1/agent/tasks
GET    /api/v1/tasks/{task_id}/memory-context
```

All endpoints require authentication. The tenant and caller permissions come from the auth token, never from the request body.

---

# 27. LangGraph State

```python
class MemoryAgentState(TypedDict):
    task_id: str
    user_query: str

    entities: list
    retrieved_memories: list

    graph_context: list
    episodic_context: list

    current_plan: dict
    task_result: dict

    candidate_memories: list
    validated_memories: list

    conflicts: list
    experiences: list

    confidence: dict
    status: str
```

Workflow:

```text
START
  |
Analyze Task
  |
Extract Entities
  |
Need Memory?
  |
  +---- NO --------------------+
  |                            |
 YES                           |
  |                            |
Retrieve Memory                |
  |                            |
Vector + Graph + Episodic      |
  |                            |
Re-rank                        |
  |                            |
Conflict Check                 |
  |                            |
Build Context                  |
  |                            |
  +-------------+--------------+
                |
                v
          Task Reasoning
                |
                v
             Execute
                |
                v
             Outcome
                |
                v
       Extract Candidate Memory
                |
                v
          Validate Memory
                |
        +-------+-------+
        |               |
      Reject          Accept
        |               |
        |               v
        |        Consolidate
        |               |
        |               v
        |          Update Graph
        |               |
        +-------+-------+
                |
                v
              END
```

---

# 28. UI Design

## Memory Dashboard (illustrative values)

```text
Enterprise Memory
──────────────────────────────────

Entities             18,421
Memories             72,830
Experiences           3,821
Conflicts                129
Pending Reviews           17

Memory Quality

High confidence       82%
Medium confidence     14%
Low confidence         4%
```

## Entity Explorer

```text
PaymentService

Current Provider:
Razorpay

Previous Provider:
Stripe

Owner:
Payments Team

Database:
Oracle

Recent Incidents:
INC-421
INC-387

Architecture Changes:
Migration-2026
```

## Memory Timeline

```text
2026-01
PaymentService → Stripe

2026-05
Migration planned

2026-08
Migration testing

2026-09-10
Stripe retired

2026-09-11
Razorpay production

2026-09-20
Architecture verified
```

---

# 29. Research Questions

## Primary

> Can confidence-aware temporal graph memory improve long-horizon enterprise-agent performance compared with vector-only memory?

## Secondary

1. Does graph memory improve multi-hop retrieval?
2. Does temporal memory reduce stale answers?
3. Does contradiction resolution improve factual consistency?
4. Does experience memory reduce repeated mistakes?
5. Does memory consolidation reduce retrieval noise?
6. How does memory quality change as sessions increase?
7. What is the cost of maintaining structured memory?
8. Can smaller models perform memory-management tasks effectively?

---

# 30. Experimental Baselines

Compare:

### Baseline A
No memory.

### Baseline B
Conversation history only.

### Baseline C
Vector memory.

### Baseline D
Vector memory + summarization.

### Baseline E
Knowledge graph memory.

### Baseline F

External memory systems (e.g., Mem0, and Zep/Graphiti as a temporal-graph reference), run on the same benchmark where feasible.

### Proposed System

```text
Vector
+
Episodic
+
Temporal Graph
+
Confidence
+
Contradiction Resolution
+
Consolidation
+
Experience Memory
```

---

# 31. Evaluation Metrics

## Retrieval

- Precision
- Recall
- F1
- MRR
- NDCG

## Temporal reasoning

- Current-fact accuracy
- Historical-fact accuracy
- Temporal relation accuracy

## Conflict resolution

- Conflict detection accuracy
- Resolution accuracy

## Memory quality

- Redundancy
- Stale-memory rate
- Unsupported-memory rate
- Contradiction rate

## Agent performance

- Task success
- Tool-use accuracy
- Planning accuracy
- Repeated-error rate

## Efficiency

- Retrieval latency
- Token consumption
- Graph traversal count
- Vector-search count
- Memory-write cost

---

# 32. Key Experiment: Repeated Mistakes

Research question:

> Can memory reduce repeated mistakes?

Without memory:

```text
Task 1 → Mistake
Task 2 → Same mistake
Task 3 → Same mistake
Task 4 → Same mistake
```

With experience memory:

```text
Task 1 → Mistake
          ↓
       Experience
          ↓
Task 2 → Avoided
Task 3 → Avoided
Task 4 → Avoided
```

Primary metric:

```text
Repeated Mistake Rate
  = tasks where a previously observed failure pattern recurs
    / tasks containing a situation matching a previously observed failure pattern
```

---

# 33. Key Experiment: Temporal Knowledge

Create a benchmark containing changing facts:

```text
Jan:
Database = PostgreSQL

Apr:
Database migration planned

Jun:
Database = Oracle

Sep:
Database = Oracle
```

Questions:

```text
What database does the system use now?

What database did it use in February?

When did the migration happen?
```

Compare vector-only memory against temporal graph memory.

---

# 34. Key Experiment: Memory Poisoning

Introduce an incorrect memory:

```text
Incorrect Memory:
PaymentService uses Stripe.
```

Actual source:

```text
PaymentService uses Razorpay.
```

Test whether the system:

1. Detects the conflict.
2. Finds authoritative evidence.
3. Updates the memory.
4. Preserves historical information.
5. Avoids repeating the incorrect fact.

---

# 35. Key Experiment: Long-Horizon Scaling

Evaluate:

```text
10 sessions
50 sessions
100 sessions
500 sessions
1000 sessions
```

Measure:

- Retrieval accuracy
- Memory redundancy
- Stale-memory rate
- Contradiction rate
- Token cost
- Task success

---

# 36. Ablation Study

Start with the complete system and remove components:

```text
Full Memory System
- Temporal metadata
- Knowledge graph
- Confidence scoring
- Contradiction resolver
- Experience memory
- Memory consolidation
```

Measure the effect of each component.

Actual values must come from experiments.

## Evaluation Methodology

- **Data**: state the benchmark source explicitly: a synthetic generator (changing facts, injected contradictions, repeated failure patterns) and, where possible, real repository and incident histories with known ground truth. Synthetic-only results must be reported as such.
- **Judging**: prefer programmatic checks against ground-truth facts. Where an LLM judge is used, validate it against a human-labelled sample and report agreement.
- **Statistics**: multiple seeds/runs, confidence intervals, and identical model and prompt budgets across baselines.
- **MVP success criteria** (fixed before running experiments): e.g., current-fact accuracy and stale-answer rate versus Baseline C on the temporal benchmark, and repeated-mistake rate versus Baseline B on the repeated-mistakes benchmark.

---

# 37. Reliability and Safety

Memory should not be treated as unquestionable truth.

For high-impact decisions:

```text
Memory
   |
   v
Evidence Verification
   |
   v
Current Source of Truth
   |
   v
Decision
```

Example:

```text
Memory says:
Oracle

Verify:
deployment.yaml
configuration
service metadata

Then:
Generate recommendation
```

---

# 38. Observability

Every memory operation should be traceable.

```text
Task
 |
 +-- Memory Retrieval
 |      |
 |      +-- Vector Search
 |      +-- Graph Search
 |      +-- Episodic Search
 |
 +-- Reasoning
 |
 +-- Action
 |
 +-- Memory Extraction
 |
 +-- Validation
 |
 +-- Consolidation
 |
 +-- Graph Update
```

Capture:

- Retrieval latency
- Number of memories retrieved
- Memory confidence
- Graph traversal depth
- Memory writes
- Conflicts detected
- Memory updates
- Token usage
- Agent outcome

---

# 39. Repository Structure

```text
enterprise-memory-agent/
│
├── agents/
│   ├── orchestrator/
│   ├── planner/
│   ├── task_agent/
│   ├── memory_retriever/
│   ├── memory_extractor/
│   ├── memory_validator/
│   ├── consolidator/
│   ├── graph_agent/
│   ├── ingestion/
│   └── reflection/
│
├── memory/
│   ├── short_term/
│   ├── episodic/
│   ├── semantic/
│   ├── procedural/
│   ├── experience/
│   ├── scoring/
│   ├── decay/
│   └── conflict_resolution/
│
├── graph/
│   ├── schema/
│   ├── entities/
│   ├── relations/
│   ├── temporal/
│   └── traversal/
│
├── retrieval/
│   ├── vector.py
│   ├── graph.py
│   ├── episodic.py
│   ├── hybrid.py
│   └── reranker.py
│
├── evaluation/
│   ├── benchmarks/
│   ├── temporal.py
│   ├── contradiction.py
│   ├── retrieval.py
│   └── long_horizon.py
│
├── api/
├── ui/
├── experiments/
├── datasets/
├── tests/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# 40. MVP Roadmap

## Phase 1 — Basic Memory

- FastAPI
- PostgreSQL
- Vector store
- Memory extraction
- Memory retrieval
- Basic agent
- Minimal validator (duplicate check, evidence required, nothing auto-trusted)
- Entity resolution (exact + alias matching)
- Evaluation harness with Baselines A–C from the start

## Phase 2 — Memory Types

Implement:

- Semantic memory
- Episodic memory
- Procedural memory
- Experience memory
- Ingestion connectors (git, configuration, incident records)

## Phase 3 — Knowledge Graph

- Entity extraction
- Relation extraction
- Graph store: PostgreSQL relations + recursive CTEs first; Neo4j only if evaluation shows a need
- Graph traversal
- Entity explorer

## Phase 4 — Temporal Memory

- valid_from
- valid_until
- observed_at
- last_verified
- historical queries

## Phase 5 — Trustworthy Memory

- Confidence scoring
- Source authority
- Contradiction detection
- Conflict resolution
- Human review

## Phase 6 — Experience Learning

- Pattern extraction
- Lesson generation
- Experience consolidation
- Repeated-error benchmark

## Phase 7 — Research Evaluation

The harness exists from Phase 1. This phase adds the full benchmark suite.

- Benchmark dataset
- Baselines
- Ablation studies
- Long-horizon experiments
- Research report

---

# 41. Future Extensions

- Cross-agent shared memory
- Permission-aware organizational memory
- Memory governance
- Active stale-memory detection
- Memory poisoning detection
- Self-improving retrieval
- Personalized memory
- Multi-agent shared knowledge graphs
- Memory compression
- Smaller-model memory managers

---

# 42. Portfolio Positioning

Moved to [Portfolio_Positioning.md](Portfolio_Positioning.md).

---

# 43. Recommended Research Contribution

The project should not claim that the general concept of agent memory is novel. Temporal knowledge-graph memory (e.g., Zep/Graphiti with bitemporal edges) and general agent memory layers (e.g., Mem0, MemGPT/Letta) already exist. The defensible contribution is the confidence, source-authority and conflict-resolution layer together with experience consolidation.

Instead, define a specific contribution:

> **Confidence-Aware Temporal Enterprise Memory with Experience Consolidation**

The proposed system combines:

```text
Persistent Memory
        +
Temporal Knowledge Graph
        +
Confidence
        +
Source Authority
        +
Contradiction Resolution
        +
Experience Learning
```

Research hypothesis:

> A memory architecture combining semantic retrieval, temporal relationships, confidence-aware updates and experience consolidation can improve long-horizon enterprise-agent reliability compared with vector-only memory.

---

# 44. Final Architecture

```text
                         USER
                           |
                           v
                  +----------------+
                  |  ORCHESTRATOR  |
                  +-------+--------+
                          |
                          v
                    +-----------+
                    | TASK AGENT|
                    +-----+-----+
                          |
                          v
                 +-------------------+
                 | MEMORY RETRIEVER  |
                 +---------+---------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
          Vector DB    Knowledge     Episodic
                       Graph          Store
              |            |            |
              +------------+------------+
                           |
                           v
                     CONTEXT BUILDER
                           |
                           v
                       REASONING
                           |
                           v
                         ACTION
                           |
                           v
                        OUTCOME
                           |
                           v
                  MEMORY EXTRACTION
                           |
                           v
                  MEMORY VALIDATION
                           |
                    +------+------+
                    |             |
                  REJECT        ACCEPT
                    |             |
                    |             v
                    |      MEMORY CONSOLIDATION
                    |             |
                    |             v
                    |       GRAPH UPDATE
                    |             |
                    +-------------+
                           |
                           v
                     FUTURE TASKS
                           |
                           +-------> RETRIEVAL
```

---

# 45. Core Thesis

The Enterprise Memory Agent is not simply an LLM with a larger context window.

It is an **evolving organizational memory layer** that allows AI agents to:

> **remember what happened, understand what is true now, understand what changed, learn from previous experiences, resolve conflicting information, and use that knowledge to perform better on future tasks.**

That is the central engineering and research problem this project investigates.
