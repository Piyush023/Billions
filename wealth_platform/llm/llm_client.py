"""
Free-tier LLM client with automatic provider fallback.

Priority order (all free):
  1. Groq        - llama-3.3-70b-versatile (14,400 req/day free)
  2. Gemini      - gemini-2.0-flash (1,500 req/day free)
  3. Ollama      - local model, unlimited, requires Ollama running

Optional paid tier (only used if explicitly requested for the final
Portfolio Manager decision and ANTHROPIC_API_KEY is set):
  4. Claude Haiku - highest quality structured decisions (~Rs.2/day)

All calls go through `LLMClient.chat()` which tries providers in order
and falls through on rate limits / errors, so a single free-tier outage
never stops the trading cycle.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger("wealth_platform.llm")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    input_chars: int = 0


@dataclass
class LLMClient:
    """Routes chat completions across free providers with fallback."""

    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    cerebras_api_key: Optional[str] = field(default_factory=lambda: os.getenv("CEREBRAS_API_KEY"))
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    use_ollama: bool = field(default_factory=lambda: os.getenv("USE_OLLAMA", "0") == "1")
    max_retries: int = 2
    request_timeout: int = 90
    rate_limit_sweeps: int = 8  # total passes over all providers when rate limited
    rate_limit_wait_s: int = 30  # free-tier limits are per-minute; 30s is a safe reset wait

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        prefer: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion, trying providers in priority order.

        prefer="anthropic" routes to Claude Haiku first (used only for the
        final Portfolio Manager decision when a key is configured).
        """
        providers = self._provider_order(prefer)
        last_error: Optional[Exception] = None

        # Outer loop: if every provider fails but at least one was only
        # rate-limited (free tiers cap tokens per MINUTE), wait for the
        # window to reset and try again instead of failing the cycle.
        for sweep in range(self.rate_limit_sweeps):
            rate_limited = False
            for provider in providers:
                for attempt in range(self.max_retries):
                    try:
                        start = time.time()
                        text = provider_dispatch[provider](self, system, user, temperature, max_tokens)
                        latency = time.time() - start
                        logger.info("LLM call ok provider=%s latency=%.1fs", provider, latency)
                        return LLMResponse(
                            text=text,
                            provider=provider,
                            model=self._model_for(provider),
                            latency_s=latency,
                            input_chars=len(system) + len(user),
                        )
                    except RateLimitError as exc:
                        logger.warning("Rate limited on %s, moving to next provider", provider)
                        last_error = exc
                        rate_limited = True
                        break  # don't retry the same rate-limited provider this sweep
                    except ProviderUnavailable as exc:
                        last_error = exc
                        break  # not configured / not running — skip silently
                    except Exception as exc:  # noqa: BLE001 - fall through to next provider
                        logger.warning("LLM call failed provider=%s attempt=%d: %s", provider, attempt + 1, exc)
                        last_error = exc
                        time.sleep(1.5 * (attempt + 1))

            if rate_limited and sweep < self.rate_limit_sweeps - 1:
                logger.info(
                    "All providers exhausted (rate limited); waiting %ds for limit window to reset (sweep %d/%d)",
                    self.rate_limit_wait_s, sweep + 1, self.rate_limit_sweeps,
                )
                time.sleep(self.rate_limit_wait_s)
            elif not rate_limited:
                break  # hard failures everywhere — waiting won't help

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def chat_json(self, system: str, user: str, max_tokens: int = 2048, prefer: Optional[str] = None) -> dict:
        """Chat call that expects a JSON object back; extracts and parses it."""
        response = self.chat(
            system=system + "\n\nRespond ONLY with a valid JSON object. No markdown fences, no prose.",
            user=user,
            temperature=0.2,
            max_tokens=max_tokens,
            prefer=prefer,
        )
        return extract_json(response.text)

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openai_compatible(self, url: str, api_key: Optional[str], model: str, provider: str,
                                system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not api_key:
            raise ProviderUnavailable(f"{provider} key not set")
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.request_timeout,
        )
        if resp.status_code == 429:
            raise RateLimitError(f"{provider} rate limited")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_groq(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        return self._call_openai_compatible(
            GROQ_URL, self.groq_api_key, DEFAULT_GROQ_MODEL, "groq", system, user, temperature, max_tokens)

    def _call_cerebras(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        return self._call_openai_compatible(
            CEREBRAS_URL, self.cerebras_api_key, DEFAULT_CEREBRAS_MODEL, "cerebras", system, user, temperature, max_tokens)

    def _call_openrouter(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        return self._call_openai_compatible(
            OPENROUTER_URL, self.openrouter_api_key, DEFAULT_OPENROUTER_MODEL, "openrouter", system, user, temperature, max_tokens)

    def _call_gemini(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not self.gemini_api_key:
            raise ProviderUnavailable("GEMINI_API_KEY not set")
        resp = requests.post(
            GEMINI_URL.format(model=DEFAULT_GEMINI_MODEL),
            params={"key": self.gemini_api_key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            },
            timeout=self.request_timeout,
        )
        if resp.status_code == 429:
            raise RateLimitError("gemini rate limited")
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if not candidates:
            raise RuntimeError("gemini returned no candidates")
        return candidates[0]["content"]["parts"][0]["text"]

    def _call_ollama(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not self.use_ollama:
            raise ProviderUnavailable("ollama disabled")
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": DEFAULT_OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False,
            },
            timeout=self.request_timeout * 2,  # local models are slower
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_anthropic(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        if not self.anthropic_api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": DEFAULT_ANTHROPIC_MODEL,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.request_timeout,
        )
        if resp.status_code == 429:
            raise RateLimitError("anthropic rate limited")
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    # ------------------------------------------------------------------

    def _provider_order(self, prefer: Optional[str]) -> List[str]:
        order = []
        if prefer == "anthropic" and self.anthropic_api_key:
            order.append("anthropic")
        if self.groq_api_key:
            order.append("groq")
        if self.cerebras_api_key:
            order.append("cerebras")
        if self.gemini_api_key:
            order.append("gemini")
        if self.openrouter_api_key:
            order.append("openrouter")
        if self.use_ollama:
            order.append("ollama")
        if not order:
            raise RuntimeError(
                "No LLM provider configured. Set GROQ_API_KEY or GEMINI_API_KEY, "
                "or run Ollama locally (https://ollama.com)."
            )
        return order

    @staticmethod
    def _model_for(provider: str) -> str:
        return {
            "groq": DEFAULT_GROQ_MODEL,
            "cerebras": DEFAULT_CEREBRAS_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
        }[provider]


class ProviderUnavailable(Exception):
    pass


class RateLimitError(Exception):
    pass


provider_dispatch = {
    "groq": LLMClient._call_groq,
    "cerebras": LLMClient._call_cerebras,
    "openrouter": LLMClient._call_openrouter,
    "gemini": LLMClient._call_gemini,
    "ollama": LLMClient._call_ollama,
    "anthropic": LLMClient._call_anthropic,
}


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in LLM response: {text[:200]}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in LLM response")
