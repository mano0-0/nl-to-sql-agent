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


USER_AGENT = "nl-to-sql-agent/0.1.0 (+https://github.com/mano0-0/nl-to-sql-agent)"


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


class OpenAICompatibleModel:
    """Any provider exposing an OpenAI-style /chat/completions endpoint.

    Groq, Together, OpenRouter, Fireworks and a local Ollama all speak this
    shape, so one implementation covers them. Uses urllib rather than adding
    an HTTP dependency for a single POST.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        provider: str = "provider",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.provider = provider

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        import json
        import urllib.error
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            # SQL generation wants the likeliest answer, not a creative one.
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode()

        headers = {
            "Content-Type": "application/json",
            # Identify the client. urllib's default ("Python-urllib/3.x") is
            # blocked outright by Cloudflare in front of some providers --
            # Groq returns 403 "error code: 1010" before the request ever
            # reaches the API.
            "User-Agent": USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code == 401:
                raise ModelError(
                    f"{self.provider} rejected the API key (401). "
                    "Check the key, or use --offline."
                ) from exc
            if exc.code == 403 and "1010" in detail:
                raise ModelError(
                    f"{self.provider} request blocked by its CDN (403/1010). "
                    "This is usually a missing or rejected User-Agent header."
                ) from exc
            if exc.code == 404 and "model" in detail:
                raise ModelError(
                    f"{self.provider} has no model {self.model!r} on this key "
                    f"(404). List what your key can use with:\n"
                    f"  curl {self.base_url}/models "
                    f'-H "Authorization: Bearer $GROQ_API_KEY"\n'
                    f"then pass one with --model."
                ) from exc
            if exc.code == 429:
                raise ModelError(
                    f"{self.provider} rate limit reached (429). "
                    "Wait a moment, or use --offline."
                ) from exc
            raise ModelError(f"{self.provider} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelError(f"could not reach {self.provider}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelError(f"{self.provider} timed out after {self.timeout}s") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(
                f"unexpected response shape from {self.provider}: {str(body)[:200]}"
            ) from exc


class GroqModel(OpenAICompatibleModel):
    """Groq's free tier. Sign up at console.groq.com/keys -- no card required."""

    BASE_URL = "https://api.groq.com/openai/v1"
    # Available on the free tier. The Llama models are Enterprise-only, so a
    # free key gets 404 model_not_found for them.
    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(
        self, api_key: Optional[str] = None, model: Optional[str] = None
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ModelError(
                "no Groq API key: set GROQ_API_KEY (free, no card, from "
                "https://console.groq.com/keys), or use --offline."
            )
        super().__init__(
            base_url=self.BASE_URL,
            model=model or os.environ.get("GROQ_MODEL", self.DEFAULT_MODEL),
            api_key=key,
            provider="Groq",
        )


class OllamaModel(OpenAICompatibleModel):
    """A model running locally via Ollama. No key, no network, no limits."""

    BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL = "qwen2.5-coder:7b"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super().__init__(
            base_url=base_url or os.environ.get("OLLAMA_HOST", self.BASE_URL),
            model=model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL),
            api_key=None,
            provider="Ollama",
        )
