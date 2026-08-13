"""
One shared LLM client for the whole project.

Both the answer generator and the LLM-judge call through here, so switching
providers means editing .env - no code changes anywhere else.

Any OpenAI-compatible endpoint works (Gemini, Groq, OpenRouter, Cerebras,
a local Ollama server). See .env.example for the base_url + model to use
for each one.
"""

from __future__ import annotations
import os
import re
import time
from dataclasses import dataclass

from openai import OpenAI

# Free tiers cap requests per minute (Gemini Flash allows ~10 RPM). Firing a
# 700-call sweep as fast as possible just triggers 429s and wastes retries, so
# space the calls out. Set LLM_MIN_INTERVAL in .env to tune: 6.5s suits 10 RPM,
# 4.0s suits 15 RPM (Flash-Lite), 0 disables throttling entirely.
_LAST_CALL_AT = 0.0


def _throttle() -> None:
    global _LAST_CALL_AT
    min_interval = float(os.environ.get("LLM_MIN_INTERVAL", "6.5"))
    if min_interval <= 0:
        return
    elapsed = time.time() - _LAST_CALL_AT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_CALL_AT = time.time()


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def make_client() -> OpenAI:
    """Builds the client from .env. Fails loudly if the key is missing."""
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY not set. Copy .env.example to .env and add your key. "
            "See that file for where to get a free one."
        )
    if not base_url:
        raise RuntimeError("LLM_BASE_URL not set. See .env.example.")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client: OpenAI,
    model: str,
    user_message: str,
    system_prompt: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.0,
    max_retries: int = 4,
) -> LLMResponse:
    """One chat completion, with token counts and latency.

    Free tiers are rate-limited (Groq allows ~30 requests/min), and a 12-config
    sweep sends a few hundred requests back to back - so a rate-limit error is
    expected, not exceptional. Retry with backoff rather than crashing a run
    that's already half done.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    for attempt in range(max_retries):
        _throttle()
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            # A 400 means the request itself is malformed or the key is bad -
            # retrying an identical request cannot help, so fail immediately
            # and surface the provider's actual message.
            status = getattr(e, "status_code", None)
            if status == 400 or "BadRequest" in type(e).__name__:
                raise RuntimeError(
                    f"Provider rejected the request (HTTP 400).\n"
                    f"  Model:    {model}\n"
                    f"  Message:  {e}\n"
                    f"Common causes: wrong LLM_MODEL name for this provider, or an "
                    f"invalid/malformed LLM_API_KEY (check for stray quotes or spaces in .env)."
                ) from e

            if attempt == max_retries - 1:
                raise

            # On a 429 the provider tells us how long to wait - honour that
            # instead of guessing, otherwise we retry too early and burn an
            # attempt for nothing.
            is_rate_limit = status == 429 or "RateLimit" in type(e).__name__
            if is_rate_limit:
                match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", str(e))
                wait = int(match.group(1)) + 2 if match else 60
                print(f"    Rate limited, waiting {wait}s...")
            else:
                wait = 2 ** attempt * 5   # 5s, 10s, 20s
                print(f"    API error ({type(e).__name__}: {e}), retrying in {wait}s...")

            time.sleep(wait)
            continue

        latency = time.time() - start
        text = response.choices[0].message.content or ""

        # Not every provider populates usage. Fall back to a rough estimate
        # (~4 chars per token) so cost numbers stay populated - but this is an
        # estimate, and the README says so.
        usage = getattr(response, "usage", None)
        if usage:
            in_tok = usage.prompt_tokens
            out_tok = usage.completion_tokens
        else:
            in_tok = len(user_message + (system_prompt or "")) // 4
            out_tok = len(text) // 4

        return LLMResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_seconds=latency,
        )

    raise RuntimeError("Unreachable")
