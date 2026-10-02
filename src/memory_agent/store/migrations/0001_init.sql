-- 0001: initial schema. Mirrors memory_agent/store/schema.py (parity is tested).
-- {{EMBED_DIM}} is substituted by the migration runner (halfvec supports up to 4000 dimensions with HNSW).
-- (the runner creates the `vector` extension in schema public before applying migrations)

CREATE TABLE IF NOT EXISTS tenants (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE, key_prefix TEXT NOT NULL,
  name TEXT, created_at TEXT COLLATE "C", revoked_at TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, entity_type TEXT NOT NULL, name TEXT NOT NULL,
  description TEXT, created_at TEXT COLLATE "C", updated_at TEXT COLLATE "C"
);
CREATE INDEX IF NOT EXISTS ix_entities_tenant ON entities (tenant_id, entity_type);
CREATE TABLE IF NOT EXISTS entity_aliases (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, entity_id TEXT NOT NULL, alias TEXT NOT NULL,
  alias_key TEXT NOT NULL, source TEXT, confidence DOUBLE PRECISION, created_at TEXT COLLATE "C",
  UNIQUE (tenant_id, alias_key)
);
CREATE TABLE IF NOT EXISTS entity_merges (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kept_entity_id TEXT NOT NULL, merged_entity_id TEXT NOT NULL,
  merged_name TEXT, moved_json TEXT, created_at TEXT COLLATE "C", reverted_at TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_type TEXT, subject_id TEXT NOT NULL,
  relation TEXT NOT NULL, object_id TEXT, object_value TEXT, content TEXT,
  confidence DOUBLE PRECISION, importance DOUBLE PRECISION, source_authority DOUBLE PRECISION,
  valid_from TEXT COLLATE "C", valid_until TEXT COLLATE "C", observed_at TEXT COLLATE "C", created_at TEXT COLLATE "C", recorded_until TEXT COLLATE "C",
  supersedes_id TEXT, last_verified TEXT COLLATE "C", last_accessed TEXT COLLATE "C", status TEXT, access_policy TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_subject ON memories (tenant_id, subject_id, relation);
CREATE INDEX IF NOT EXISTS ix_mem_object ON memories (tenant_id, object_id);
CREATE INDEX IF NOT EXISTS ix_mem_status ON memories (tenant_id, status);
CREATE OR REPLACE VIEW relations AS
  SELECT id, tenant_id, subject_id AS source_entity_id, relation, object_id AS target_entity_id,
         confidence, valid_from, valid_until, recorded_until, id AS source_memory_id
  FROM memories WHERE object_id IS NOT NULL;
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_id TEXT NOT NULL, source_type TEXT,
  source_uri TEXT, excerpt TEXT, source_authority DOUBLE PRECISION, observed_at TEXT COLLATE "C", access_policy TEXT
);
CREATE INDEX IF NOT EXISTS ix_evidence_memory ON evidence (memory_id);
CREATE TABLE IF NOT EXISTS reviews (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, memory_id TEXT, candidate_json TEXT,
  reason TEXT, status TEXT DEFAULT 'pending', reviewer TEXT, note TEXT,
  created_at TEXT COLLATE "C", decided_at TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, uri TEXT, content_hash TEXT NOT NULL,
  source_type TEXT, doc_date TEXT COLLATE "C", ingested_at TEXT COLLATE "C", UNIQUE (tenant_id, content_hash)
);
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, task_id TEXT, summary TEXT, outcome TEXT,
  timestamp TEXT COLLATE "C", signature TEXT, conditions TEXT, counter_of TEXT, ordinal BIGINT
);
CREATE TABLE IF NOT EXISTS episode_entities (episode_id TEXT NOT NULL, entity_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiences (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, status TEXT, pattern TEXT, lesson TEXT,
  evidence_count INTEGER, confidence DOUBLE PRECISION, conditions TEXT, recommended_action TEXT,
  created_at TEXT COLLATE "C", last_verified TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS experience_evidence (experience_id TEXT NOT NULL, episode_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, task TEXT NOT NULL, allowed_steps TEXT, plan TEXT,
  experiences TEXT, context TEXT, status TEXT, outcome TEXT, error TEXT, episode_id TEXT,
  created_at TEXT COLLATE "C", updated_at TEXT COLLATE "C"
);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
  error TEXT, result TEXT, run_after TEXT COLLATE "C", locked_at TEXT COLLATE "C", locked_by TEXT,
  created_at TEXT COLLATE "C", started_at TEXT COLLATE "C", finished_at TEXT COLLATE "C"
);
CREATE INDEX IF NOT EXISTS ix_jobs_claim ON jobs (status, run_after);
CREATE INDEX IF NOT EXISTS ix_jobs_tenant ON jobs (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS vectors (
  kind TEXT NOT NULL, ref_id TEXT NOT NULL, tenant_id TEXT NOT NULL, vec halfvec({{EMBED_DIM}}) NOT NULL,
  PRIMARY KEY (kind, ref_id)
);
CREATE INDEX IF NOT EXISTS ix_vectors_tenant ON vectors (tenant_id, kind);
CREATE INDEX IF NOT EXISTS ix_vectors_hnsw ON vectors USING hnsw (vec halfvec_cosine_ops);
