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
def ctx(clock, script):
    def handler(prompt, schema):
        for marker, facts in script.items():
            if marker in prompt:
                return ExtractionResult(facts=facts)
        return ExtractionResult()

    return Context(MemoryStore(), FakeLLM(handler), FakeEmbedder(), Settings(), clock)
