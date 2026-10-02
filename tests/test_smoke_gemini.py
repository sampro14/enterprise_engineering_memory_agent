"""Live provider smoke test through the factory. Skipped unless RUN_LIVE=1 (costs API calls)."""
import os

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.live, pytest.mark.skipif(os.getenv("RUN_LIVE") != "1", reason="live test")]


class Answer(BaseModel):
    provider: str


def test_gemini_roundtrip_via_factory(tmp_path):
    from memory_agent.config import Settings
    from memory_agent.llm.factory import create_embedder, create_llm

    s = Settings(llm_provider="gemini", embed_provider="gemini", cache_path=str(tmp_path / "c.db"))
    a = create_llm(s).generate_json("The payment service uses Razorpay. Which provider? Return provider.", Answer)
    assert "razorpay" in a.provider.lower()
    e = create_embedder(s).embed(["hello", "hi there"])
    assert e.shape == (2, 3072) and abs((e[0] ** 2).sum() - 1) < 1e-3
