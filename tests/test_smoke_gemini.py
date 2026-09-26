"""Live Gemini smoke test. Skipped unless RUN_LIVE=1 (costs API calls)."""
import os

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="live test")


class Answer(BaseModel):
    provider: str


def test_gemini_roundtrip(tmp_path):
    from memory_agent.config import Settings
    from memory_agent.llm.cache import DiskCache
    from memory_agent.llm.gemini import GeminiEmbedder, GeminiLLM

    cache = DiskCache(str(tmp_path / "c.db"))
    llm = GeminiLLM(Settings(), cache)
    a = llm.generate_json("The payment service uses Razorpay. Which provider? Return provider.", Answer)
    assert "razorpay" in a.provider.lower()
    e = GeminiEmbedder(Settings(), cache).embed(["hello", "hi there"])
    assert e.shape[0] == 2 and abs((e[0] ** 2).sum() - 1) < 1e-3
