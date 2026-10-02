"""Anthropic (Claude) adapter. LLM only: Anthropic has no embeddings API.

NOT live-tested in this repo (no ANTHROPIC_API_KEY was available); covered by mocked contract tests.
The installed SDK exposes no `temperature` parameter on messages.create/parse, so output is not pinned to
temperature 0 for this provider.
"""
from __future__ import annotations

import anthropic
from pydantic import ValidationError

from ..base import StructuredOutputError, T


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096, client=None):
        self.model, self.max_tokens = model, max_tokens
        self.client = client or anthropic.Anthropic(api_key=api_key)

    def _kwargs(self, prompt: str, system: str | None) -> dict:
        kw: dict = {"model": self.model, "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
        if system:
            kw["system"] = system
        return kw

    def generate(self, prompt: str, system: str | None = None) -> str:
        r = self.client.messages.create(**self._kwargs(prompt, system))
        return "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        r = self.client.messages.parse(output_format=schema, **self._kwargs(prompt, system))
        for block in r.content:
            parsed = getattr(block, "parsed_output", None)
            if parsed is not None:
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        try:
            return schema.model_validate_json(text)
        except (ValidationError, ValueError) as e:
            raise StructuredOutputError(f"anthropic returned output that does not match {schema.__name__}: {e}") from e
