import os
import uuid

import pytest

from memory_agent.config import Settings
from memory_agent.context import Context
from memory_agent.llm.fake import FakeEmbedder, FakeLLM
from memory_agent.models import ExtractedFact, ExtractionResult
from memory_agent.store.sqlite import MemoryStore


def fact(subject, relation, obj, evidence, valid_from=None, valid_until=None, **kw):
    return ExtractedFact(subject=subject, subject_type="service", relation=relation, object=obj,
                         evidence=evidence, valid_from=valid_from, valid_until=valid_until, **kw)


class Clock:
    def __init__(self, d="2025-01-15"):
        self.d = d

    def __call__(self):
        return self.d


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def script():
    """Map a marker string found in the prompt -> facts the fake extractor returns."""
    return {}


@pytest.fixture
def ctx(clock, script, backend):
    def handler(prompt, schema):
        for marker, facts in script.items():
            if marker in prompt:
                return ExtractionResult(facts=facts)
        return ExtractionResult()

    return Context(backend, FakeLLM(handler), FakeEmbedder(), Settings(), clock)


# ---------------------------------------------------------------- storage backends (SQLite always, Postgres if DATABASE_URL)
EMBED_DIM = 256  # matches FakeEmbedder's default


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.integration)])
def backend(request):
    """A tenant-bound store on each backend. Postgres tests use a throwaway schema, dropped afterwards."""
    if request.param == "sqlite":
        st = MemoryStore()
        yield st
        st.close()
        return
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("set DATABASE_URL to run Postgres integration tests")
    import psycopg

    from memory_agent.store.migrate import apply_migrations
    from memory_agent.store.postgres import PostgresStore

    schema = "t_" + uuid.uuid4().hex[:10]
    apply_migrations(url, EMBED_DIM, schema)
    st = PostgresStore(url, "tenantA", schema=schema, embed_dim=EMBED_DIM, pool_size=8)
    try:
        yield st
    finally:
        st.close()
        with psycopg.connect(url, autocommit=True) as c:
            c.execute(f'DROP SCHEMA "{schema}" CASCADE')
