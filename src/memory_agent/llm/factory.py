"""Provider factory: `create_llm(settings)` / `create_embedder(settings)`.

Providers are selected by name (LLM_PROVIDER / EMBED_PROVIDER) and are independent, so e.g. Claude for
generation can be combined with Gemini or OpenAI embeddings. Retries and the on-disk cache are applied here,
once, around whichever provider is chosen (cache outermost so hits skip retries).

Add a provider by writing an adapter and registering a builder:

    @register_llm("myprovider")
    def _build(s: Settings, model: str) -> LLMClient: ...
"""
from __future__ import annotations

from collections.abc import Callable

from ..config import Settings, get_settings
from .base import Embedder, LLMClient, ProviderConfigError
from .cache import DiskCache
from .decorators import CachedEmbedder, CachedLLM, RetryingEmbedder, RetryingLLM
from .fake import FakeEmbedder, FakeLLM, default_handler

DEFAULT_LLM_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-5", "openai": "gpt-4.1-mini", "fake": "fake"}
DEFAULT_EMBED_MODELS = {"gemini": "gemini-embedding-001", "openai": "text-embedding-3-large", "fake": "fake-hash"}

LLMBuilder = Callable[[Settings, str], LLMClient]
EmbedBuilder = Callable[[Settings, str], Embedder]
_LLMS: dict[str, LLMBuilder] = {}
_EMBEDDERS: dict[str, EmbedBuilder] = {}
_caches: dict[str, DiskCache] = {}


def register_llm(name: str) -> Callable[[LLMBuilder], LLMBuilder]:
    def deco(fn: LLMBuilder) -> LLMBuilder:
        _LLMS[name] = fn
        return fn
    return deco


def register_embedder(name: str) -> Callable[[EmbedBuilder], EmbedBuilder]:
    def deco(fn: EmbedBuilder) -> EmbedBuilder:
        _EMBEDDERS[name] = fn
        return fn
    return deco


def available_llm_providers() -> list[str]:
    return sorted(_LLMS)


def available_embed_providers() -> list[str]:
    return sorted(_EMBEDDERS)


def _key(value: str, env: str, provider: str) -> str:
    if not value:
        raise ProviderConfigError(f"{env} is not set (required by provider '{provider}')")
    return value


# ---- builders (SDK imports are deferred so an unused provider never has to be importable) ----------------------------
@register_llm("gemini")
def _gemini_llm(s: Settings, model: str) -> LLMClient:
    from .providers.gemini import GeminiLLM
    return GeminiLLM(_key(s.google_api_key, "GOOGLE_API_KEY", "gemini"), model)


@register_llm("anthropic")
def _anthropic_llm(s: Settings, model: str) -> LLMClient:
    from .providers.anthropic_provider import AnthropicLLM
    return AnthropicLLM(_key(s.anthropic_api_key, "ANTHROPIC_API_KEY", "anthropic"), model)


@register_llm("openai")
def _openai_llm(s: Settings, model: str) -> LLMClient:
    from .providers.openai_provider import OpenAILLM
    return OpenAILLM(_key(s.openai_api_key, "OPENAI_API_KEY", "openai"), model)


@register_llm("fake")
def _fake_llm(s: Settings, model: str) -> LLMClient:
    return FakeLLM(default_handler, model=model)


@register_embedder("gemini")
def _gemini_embedder(s: Settings, model: str) -> Embedder:
    from .providers.gemini import GeminiEmbedder
    return GeminiEmbedder(_key(s.google_api_key, "GOOGLE_API_KEY", "gemini"), model, s.embed_dim)


@register_embedder("openai")
def _openai_embedder(s: Settings, model: str) -> Embedder:
    from .providers.openai_provider import OpenAIEmbedder
    return OpenAIEmbedder(_key(s.openai_api_key, "OPENAI_API_KEY", "openai"), model, s.embed_dim)


@register_embedder("fake")
def _fake_embedder(s: Settings, model: str) -> Embedder:
    return FakeEmbedder(s.embed_dim, model=model)


# ---- public API ---------------------------------------------------------------------------------------------------
def get_cache(settings: Settings) -> DiskCache | None:
    if not settings.cache_path:
        return None
    if settings.cache_path not in _caches:
        _caches[settings.cache_path] = DiskCache(settings.cache_path)
    return _caches[settings.cache_path]


def create_llm(settings: Settings | None = None, *, retry: bool = True, cache: bool = True) -> LLMClient:
    s = settings or get_settings()
    name = s.llm_provider
    if name not in _LLMS:
        raise ProviderConfigError(f"unknown LLM_PROVIDER '{name}'; available: {', '.join(available_llm_providers())}")
    model = s.llm_model or DEFAULT_LLM_MODELS.get(name, "")
    if not model:
        raise ProviderConfigError(f"LLM_MODEL is required for provider '{name}'")
    client: LLMClient = _LLMS[name](s, model)
    if name == "fake":
        return client
    if retry:
        client = RetryingLLM(client)
    if cache and (c := get_cache(s)) is not None:
        client = CachedLLM(client, c)
    return client


def create_embedder(settings: Settings | None = None, *, retry: bool = True, cache: bool = True) -> Embedder:
    s = settings or get_settings()
    name = s.embed_provider
    if name not in _EMBEDDERS:
        raise ProviderConfigError(
            f"unknown EMBED_PROVIDER '{name}'; available: {', '.join(available_embed_providers())} "
            "(Anthropic has no embeddings API)"
        )
    model = s.embed_model or DEFAULT_EMBED_MODELS.get(name, "")
    if not model:
        raise ProviderConfigError(f"EMBED_MODEL is required for provider '{name}'")
    emb: Embedder = _EMBEDDERS[name](s, model)
    if emb.dim != s.embed_dim:
        raise ProviderConfigError(f"embedder dim {emb.dim} != EMBED_DIM {s.embed_dim}")
    if name == "fake":
        return emb
    if retry:
        emb = RetryingEmbedder(emb)
    if cache and (c := get_cache(s)) is not None:
        emb = CachedEmbedder(emb, c)
    return emb
