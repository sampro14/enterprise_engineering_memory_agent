"""Provider factory + adapter contract tests. All SDK clients are mocked: no network, no keys."""
import types

import numpy as np
import pytest
from pydantic import BaseModel

from memory_agent.config import Settings
from memory_agent.llm.base import (
    LLMError,
    ProviderConfigError,
    StructuredOutputError,
    TransientLLMError,
)
from memory_agent.llm.cache import DiskCache
from memory_agent.llm.decorators import (
    CachedEmbedder,
    CachedLLM,
    RetryingEmbedder,
    RetryingLLM,
    default_is_transient,
)
from memory_agent.llm.factory import create_embedder, create_llm
from memory_agent.llm.providers.anthropic_provider import AnthropicLLM
from memory_agent.llm.providers.gemini import GeminiEmbedder, GeminiLLM
from memory_agent.llm.providers.openai_provider import OpenAIEmbedder, OpenAILLM


class Out(BaseModel):
    provider: str


def ns(**kw):
    return types.SimpleNamespace(**kw)


# ---------------------------------------------------------------- mocked SDK clients
class GeminiClient:
    def __init__(self, text='{"provider": "razorpay"}', dim=3072):
        self.text, self.dim, self.calls = text, dim, []
        self.models = ns(generate_content=self._gen, embed_content=self._emb)

    def _gen(self, model, contents, config):
        self.calls.append(("gen", model, contents, config.system_instruction))
        return ns(text=self.text)

    def _emb(self, model, contents, config=None):
        self.calls.append(("emb", model, len(contents)))
        return ns(embeddings=[ns(values=list(np.arange(1, self.dim + 1, dtype=float) * (i + 1))) for i in range(len(contents))])


class AnthropicClient:
    def __init__(self, parsed=True, text='{"provider": "razorpay"}'):
        self.parsed, self.text, self.calls = parsed, text, []
        self.messages = ns(create=self._create, parse=self._parse)

    def _create(self, **kw):
        self.calls.append(("create", kw))
        return ns(content=[ns(type="text", text="hello")])

    def _parse(self, **kw):
        self.calls.append(("parse", kw))
        block = ns(type="text", text=self.text, parsed_output=kw["output_format"](provider="razorpay") if self.parsed else None)
        return ns(content=[block])


class OpenAIClient:
    def __init__(self, parsed=True, refusal=None, content='{"provider": "razorpay"}', dim=3072):
        self.parsed, self.refusal, self.content, self.dim, self.calls = parsed, refusal, content, dim, []
        self.chat = ns(completions=ns(create=self._create, parse=self._parse))
        self.embeddings = ns(create=self._emb)

    def _create(self, model, messages):
        self.calls.append(("create", messages))
        return ns(choices=[ns(message=ns(content="hello"))])

    def _parse(self, model, messages, response_format):
        self.calls.append(("parse", messages))
        msg = ns(parsed=response_format(provider="razorpay") if self.parsed else None, refusal=self.refusal, content=self.content)
        return ns(choices=[ns(message=msg)])

    def _emb(self, model, input, dimensions):
        self.calls.append(("emb", dimensions))
        data = [ns(index=i, embedding=list(np.arange(1, dimensions + 1, dtype=float) * (i + 1))) for i in reversed(range(len(input)))]
        return ns(data=data)


LLM_CASES = {
    "gemini": lambda: GeminiLLM("k", "m", client=GeminiClient()),
    "anthropic": lambda: AnthropicLLM("k", "m", client=AnthropicClient()),
    "openai": lambda: OpenAILLM("k", "m", client=OpenAIClient()),
}


# ---------------------------------------------------------------- adapter contract
@pytest.mark.parametrize("name", LLM_CASES)
def test_llm_contract_json_and_text(name):
    llm = LLM_CASES[name]()
    out = llm.generate_json("Which provider?", Out, system="be brief")
    assert isinstance(out, Out) and out.provider == "razorpay"
    assert isinstance(llm.generate("hi", system="be brief"), str)
    assert llm.model == "m"


def test_system_prompt_is_forwarded():
    g = GeminiClient()
    GeminiLLM("k", "m", client=g).generate_json("q", Out, system="SYS")
    assert g.calls[0][3] == "SYS"
    a = AnthropicClient()
    AnthropicLLM("k", "m", client=a).generate_json("q", Out, system="SYS")
    assert a.calls[0][1]["system"] == "SYS" and a.calls[0][1]["output_format"] is Out
    o = OpenAIClient()
    OpenAILLM("k", "m", client=o).generate_json("q", Out, system="SYS")
    assert o.calls[0][1][0] == {"role": "system", "content": "SYS"}


def test_bad_structured_output_raises_clear_error():
    with pytest.raises(StructuredOutputError):
        GeminiLLM("k", "m", client=GeminiClient(text="not json")).generate_json("q", Out)
    with pytest.raises(StructuredOutputError):
        AnthropicLLM("k", "m", client=AnthropicClient(parsed=False, text="nope")).generate_json("q", Out)
    with pytest.raises(StructuredOutputError):
        OpenAILLM("k", "m", client=OpenAIClient(parsed=False, content="nope")).generate_json("q", Out)
    with pytest.raises(StructuredOutputError, match="refused"):
        OpenAILLM("k", "m", client=OpenAIClient(parsed=False, refusal="policy")).generate_json("q", Out)


def test_anthropic_falls_back_to_text_json_when_no_parsed_output():
    out = AnthropicLLM("k", "m", client=AnthropicClient(parsed=False)).generate_json("q", Out)
    assert out.provider == "razorpay"


@pytest.mark.parametrize("make", [
    lambda: GeminiEmbedder("k", "m", 3072, client=GeminiClient()),
    lambda: OpenAIEmbedder("k", "m", 3072, client=OpenAIClient()),
])
def test_embedder_contract_normalized_and_ordered(make):
    e = make()
    v = e.embed(["a", "b", "c"])
    assert v.shape == (3, 3072) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert e.dim == 3072


def test_embedder_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match="EMBED_DIM"):
        GeminiEmbedder("k", "m", 3072, client=GeminiClient(dim=768)).embed(["a"])


# ---------------------------------------------------------------- decorators
class Flaky:
    model = "m"

    def __init__(self, fail_with, times=2):
        self.fail_with, self.times, self.n = fail_with, times, 0

    def generate(self, prompt, system=None):
        self.n += 1
        if self.n <= self.times:
            raise self.fail_with
        return "ok"

    def generate_json(self, prompt, schema, system=None):
        return schema(provider="x")


def exc_with(**kw):
    e = Exception("x")
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def test_retry_only_transient_errors():
    sleeps = []
    err429 = type("RateLimitError", (Exception,), {})("slow down")
    f = Flaky(err429)
    assert RetryingLLM(f, attempts=5, base_delay=0.1, sleep=sleeps.append).generate("p") == "ok"
    assert f.n == 3 and sleeps == [0.1, 0.2]
    boom = Flaky(ValueError("bad request"))
    with pytest.raises(ValueError):
        RetryingLLM(boom, sleep=sleeps.append).generate("p")
    assert boom.n == 1  # non-transient: no retry
    always = Flaky(TransientLLMError("down"), times=99)
    with pytest.raises(TransientLLMError):
        RetryingLLM(always, attempts=3, sleep=lambda _s: None).generate("p")
    assert always.n == 3


def test_is_transient_by_status_code_and_name():
    assert default_is_transient(exc_with(status_code=429))
    assert default_is_transient(exc_with(code=503))
    assert not default_is_transient(exc_with(status_code=400))
    assert not default_is_transient(KeyError("x"))


def test_cached_llm_skips_inner_on_hit_and_matches_poc_key_layout(tmp_path):
    cache = DiskCache(str(tmp_path / "c.db"))
    g = GeminiClient()
    llm = CachedLLM(GeminiLLM("k", "gemini-2.5-flash", client=g), cache)
    a = llm.generate_json("q", Out, system="s")
    b = llm.generate_json("q", Out, system="s")
    assert a == b and len(g.calls) == 1
    # key layout identical to the PoC, so previously cached eval runs stay valid
    assert cache.get(DiskCache.key("json", "gemini-2.5-flash", "s", "Out", "q")) is not None


def test_cached_embedder_only_embeds_misses(tmp_path):
    cache = DiskCache(str(tmp_path / "c.db"))
    g = GeminiClient()
    emb = CachedEmbedder(GeminiEmbedder("k", "m", 3072, client=g), cache, batch_size=2)
    emb.embed(["a", "b"])
    emb.embed(["a", "b", "c"])
    assert [c for c in g.calls if c[0] == "emb"] == [("emb", "m", 2), ("emb", "m", 1)]


def test_retrying_embedder_retries():
    class E:
        model, dim, n = "m", 4, 0

        def embed(self, texts):
            self.n += 1
            if self.n < 2:
                raise TransientLLMError("x")
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    e = E()
    assert RetryingEmbedder(e, sleep=lambda _s: None).embed(["a"]).shape == (1, 4)
    assert e.n == 2


# ---------------------------------------------------------------- factory
def settings(tmp_path, **kw):
    base = dict(cache_path=str(tmp_path / "c.db"), google_api_key="g", anthropic_api_key="a", openai_api_key="o")
    return Settings(**{**base, **kw})


@pytest.mark.parametrize("provider,cls,model", [
    ("gemini", GeminiLLM, "gemini-2.5-flash"),
    ("anthropic", AnthropicLLM, "claude-sonnet-5"),
    ("openai", OpenAILLM, "gpt-4.1-mini"),
])
def test_factory_builds_each_llm_with_default_model_and_wrappers(tmp_path, provider, cls, model):
    llm = create_llm(settings(tmp_path, llm_provider=provider, llm_model=""))
    assert isinstance(llm, CachedLLM) and isinstance(llm.inner, RetryingLLM) and isinstance(llm.inner.inner, cls)
    assert llm.model == model


def test_factory_model_override_and_no_cache_no_retry(tmp_path):
    llm = create_llm(settings(tmp_path, llm_provider="openai", llm_model="gpt-x", cache_path=""), retry=False)
    assert isinstance(llm, OpenAILLM) and llm.model == "gpt-x"


def test_claude_for_generation_plus_openai_embeddings(tmp_path):
    s = settings(tmp_path, llm_provider="anthropic", embed_provider="openai", embed_model="")
    assert isinstance(create_llm(s).inner.inner, AnthropicLLM)
    e = create_embedder(s)
    assert isinstance(e.inner.inner, OpenAIEmbedder) and e.dim == 3072


def test_factory_errors_are_clear(tmp_path):
    with pytest.raises(ProviderConfigError, match="unknown LLM_PROVIDER 'nope'.*gemini"):
        create_llm(settings(tmp_path, llm_provider="nope"))
    with pytest.raises(ProviderConfigError, match="Anthropic has no embeddings"):
        create_embedder(settings(tmp_path, embed_provider="anthropic"))
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        create_llm(settings(tmp_path, llm_provider="anthropic", anthropic_api_key=""))
    with pytest.raises(ProviderConfigError, match="GOOGLE_API_KEY"):
        create_embedder(settings(tmp_path, embed_provider="gemini", google_api_key=""))


def test_fake_provider_needs_no_keys_and_is_unwrapped(tmp_path):
    s = Settings(llm_provider="fake", embed_provider="fake", embed_dim=64, cache_path=str(tmp_path / "c.db"),
                 llm_model="", embed_model="")
    llm, emb = create_llm(s), create_embedder(s)
    assert llm.model == "fake" and emb.embed(["hello world"]).shape == (1, 64)
    with pytest.raises(LLMError, match="no scripted response"):
        llm.generate_json("x", Out)  # required field, no default: explicit failure instead of silent garbage
