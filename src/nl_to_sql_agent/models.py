"""The language model boundary.

The agent depends on the small :class:`LanguageModel` protocol below rather
than on any particular provider. Two consequences:

* the pipeline can be tested end to end with no API key and no network, using
  :class:`ScriptedModel`;
* swapping provider is a constructor change, not a rewrite.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Protocol, Sequence


class ModelError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


class LanguageModel(Protocol):
    """Anything that can turn a prompt into text."""

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        ...


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response.

    Models wrap SQL in fenced code blocks about as often as not, and sometimes
    add a sentence of commentary. Taking the fenced block when present, and the
    whole response otherwise, handles both without a brittle parser.
    """
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    sql = fenced.group(1) if fenced else text
    return sql.strip().rstrip(";").strip()


class ScriptedModel:
    """A model that replays prepared answers. For tests and offline demos.

    Keyed by substring match on the user prompt so a test can say "when asked
    about revenue, return this SQL" without reproducing the prompt verbatim.
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default: str = "SELECT 1",
        answers: Optional[Dict[str, str]] = None,
        default_answer: str = "Here are the results.",
    ) -> None:
        self.responses = responses or {}      # consulted when writing SQL
        self.answers = answers or {}          # consulted when explaining
        self.default = default
        self.default_answer = default_answer
        self.calls: List[Dict[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        self.calls.append({"system": system, "user": user})
        # Route on the system prompt: the explanation call carries the question
        # text too, so matching on `user` alone would replay the SQL response.
        explaining = "explain query results" in system.lower()
        table = self.answers if explaining else self.responses
        haystack = user.lower()
        for needle, response in table.items():
            if needle.lower() in haystack:
                return response
        return self.default_answer if explaining else self.default


class AnthropicModel:
    """Claude, via the official SDK.

    Imported lazily so the package stays usable -- and testable -- without the
    ``anthropic`` dependency installed.
    """

    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelError(
                "the `anthropic` package is required for live answers: "
                "pip install 'nl-to-sql-agent[anthropic]'"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError(
                "no API key: set ANTHROPIC_API_KEY, or use --offline to run "
                "against the scripted model."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ModelError(f"model request failed: {exc}") from exc

        # Select text blocks by type rather than by position: a thinking-enabled
        # model emits a reasoning block first, so content[0] is not reliably text.
        parts: Sequence[str] = [
            block.text for block in message.content
            if getattr(block, "type", None) == "text"
        ]
        if not parts:
            raise ModelError("model returned no text content")
        return "\n".join(parts)
