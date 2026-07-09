"""Deterministic offline backend.

The pipeline needs to run end to end with no API key. Each calling layer computes
a deterministic best-effort
answer from the data it already holds (regex number-finding, lexical scan,
template synthesis) and passes it as `context["mock"]`. The live model ignores
that hint and reasons for real; the mock simply validates it back into the
requested schema. Intelligence lives in the layer that owns the data — never
hidden in the mock.
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockBackend:
    def structured(self, prompt: str, schema: Type[T], context: dict) -> tuple[T, int]:
        hint = context.get("mock")
        if hint is not None:
            return schema.model_validate(hint), 0
        # No precomputed answer: build the schema's zero value so callers never crash.
        return schema.model_construct(), 0

    def complete(self, prompt: str, context: dict) -> tuple[str, int]:
        text = context.get("mock_text")
        if text is None:
            text = "[mock] " + prompt[:200]
        return text, _estimate_tokens(text)
