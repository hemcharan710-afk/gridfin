"""LLM access layer.

Structured output is done with a forced single-schema tool call on the provider
SDK, so responses parse straight into Pydantic models. Keeping it on the raw SDK
keeps dependencies light and leaves a clean seam for a mock backend, so the whole
pipeline runs offline with no key.

Two providers are supported, selected by `llm_provider` ("anthropic" | "openai").
Either way, the numeric store and calc engine still own every number.

Model routing lives here: callers ask for tier "small" or "large".
"""

from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel

from gridfin.config import Settings, get_settings
from gridfin.llm.mock import MockBackend

T = TypeVar("T", bound=BaseModel)

Tier = str  # "small" | "large"


class LLMResult(BaseModel):
    text: str
    tokens: int = 0
    model: str = ""


def _certifi_http_client():
    """Build an httpx client pinned to certifi's CA bundle.

    macOS Python builds often don't point at a valid system CA store, which makes
    the provider SDKs fail with APIConnectionError (SSL: CERTIFICATE_VERIFY_FAILED)
    even though `curl` works. Pinning to certifi makes TLS verification
    deterministic regardless of the host's SSL configuration. Returns None if the
    deps aren't importable, so the SDK falls back to its own defaults."""
    try:
        import ssl

        import certifi
        import httpx

        ctx = ssl.create_default_context(cafile=certifi.where())
        return httpx.AsyncClient(verify=ctx, timeout=httpx.Timeout(60.0, connect=15.0))
    except Exception:  # pragma: no cover - defensive
        return None


class LLM:
    """Tiered structured-output client with an offline mock fallback."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._mock = MockBackend()
        self._client = None
        self.provider = self.settings.provider
        if self.settings.use_live_llm:
            try:
                http_client = _certifi_http_client()
                if self.provider == "openai":
                    from openai import AsyncOpenAI

                    kw = {"http_client": http_client} if http_client is not None else {}
                    self._client = AsyncOpenAI(
                        api_key=(self.settings.openai_api_key or "").strip() or None, **kw
                    )
                else:
                    from anthropic import AsyncAnthropic

                    kw = {"http_client": http_client} if http_client is not None else {}
                    self._client = AsyncAnthropic(
                        api_key=(self.settings.anthropic_api_key or "").strip() or None, **kw
                    )
            except Exception:  # pragma: no cover - defensive: fall back to mock
                self._client = None

    @property
    def live(self) -> bool:
        return self._client is not None

    def _model_for(self, tier: Tier) -> str:
        if self.provider == "openai":
            return (
                self.settings.openai_model_large
                if tier == "large"
                else self.settings.openai_model_small
            )
        return self.settings.model_large if tier == "large" else self.settings.model_small

    # -- structured output -------------------------------------------------
    async def structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        tier: Tier = "small",
        system: str = "",
        context: dict | None = None,
    ) -> tuple[T, int]:
        """Return a validated `schema` instance plus tokens spent.

        `context` carries structured hints the mock backend uses to produce a
        deterministic, sensible answer when there is no live model.
        """
        if not self.live:
            obj, tokens = self._mock.structured(prompt, schema, context or {})
            return obj, tokens

        if self.provider == "openai":
            return await self._structured_openai(prompt, schema, tier=tier, system=system)

        tool = {
            "name": "emit",
            "description": f"Emit a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        msg = await self._client.messages.create(
            model=self._model_for(tier),
            max_tokens=2048,
            system=system or "You are GridFin, a precise financial-document analyst.",
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": prompt}],
        )
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        payload = next(b.input for b in msg.content if b.type == "tool_use")
        return schema.model_validate(payload), tokens

    async def _structured_openai(
        self, prompt: str, schema: Type[T], *, tier: Tier, system: str
    ) -> tuple[T, int]:
        """OpenAI structured output via a forced single-function tool call."""
        tool = {
            "type": "function",
            "function": {
                "name": "emit",
                "description": f"Emit a {schema.__name__} object.",
                "parameters": schema.model_json_schema(),
            },
        }
        resp = await self._client.chat.completions.create(
            model=self._model_for(tier),
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": system or "You are GridFin, a precise financial-document analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "emit"}},
        )
        msg = resp.choices[0].message
        payload = json.loads(msg.tool_calls[0].function.arguments)
        tokens = resp.usage.total_tokens if resp.usage else 0
        return schema.model_validate(payload), tokens

    # -- free-form completion ---------------------------------------------
    async def complete(
        self, prompt: str, *, tier: Tier = "small", system: str = "", context: dict | None = None
    ) -> LLMResult:
        if not self.live:
            text, tokens = self._mock.complete(prompt, context or {})
            return LLMResult(text=text, tokens=tokens, model="mock")

        if self.provider == "openai":
            resp = await self._client.chat.completions.create(
                model=self._model_for(tier),
                max_tokens=2048,
                messages=[
                    {
                        "role": "system",
                        "content": system
                        or "You are GridFin, a precise financial-document analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return LLMResult(
                text=resp.choices[0].message.content or "",
                tokens=resp.usage.total_tokens if resp.usage else 0,
                model=self._model_for(tier),
            )

        msg = await self._client.messages.create(
            model=self._model_for(tier),
            max_tokens=2048,
            system=system or "You are GridFin, a precise financial-document analyst.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(
            text=text,
            tokens=msg.usage.input_tokens + msg.usage.output_tokens,
            model=self._model_for(tier),
        )


_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM
