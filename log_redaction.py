"""
Log redaction helpers (added by local patch, not part of the official template).

Masks key/token-like strings before they reach log output:
  - OpenAI/OpenRouter/Anthropic style keys:   sk-..., sk-or-v1-..., sk-ant-...
  - HTTP auth headers:                        "Bearer <x>", "Token <x>"
  - key=value / key: value pairs whose key looks secret (api_key, secret, token, password)
  - long hex strings (>= 32 chars)
  - long base64/base64url-like strings (>= 40 chars, mixed upper/lower/digit)
  - literal values of known secret environment variables (METACULUS_TOKEN, OPENROUTER_API_KEY, ...)

Pure standard library. No network access.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import Iterable

REDACTED = "[REDACTED]"

# Environment variables whose *values* must never appear in logs.
SECRET_ENV_VARS: tuple[str, ...] = (
    "METACULUS_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ASKNEWS_CLIENT_ID",
    "ASKNEWS_SECRET",
    "PERPLEXITY_API_KEY",
    "EXA_API_KEY",
    "LIGHTNINGROD_API_KEY",
)
# Values shorter than this are not literally replaced (avoid masking e.g. "1" or "true").
_MIN_ENV_VALUE_LEN = 8

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # sk-..., sk-or-v1-..., sk-ant-api03-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), REDACTED),
    # Authorization headers: keep the scheme word, hide the credential.
    (
        re.compile(r"(?i)\b(Bearer|Token)(\s+)[A-Za-z0-9._~+/=\-]{8,}"),
        r"\1\2" + REDACTED,
    ),
    # key=value / "key": "value" where the key name looks secret.
    (
        re.compile(
            r"(?i)(\b[A-Za-z0-9_\-]*(?:api[_\-]?key|secret|token|password|passwd)\b"
            r"[\"']?\s*[:=]\s*[\"']?)([^\s\"',;}]{6,})"
        ),
        r"\1" + REDACTED,
    ),
    # Long hex strings (hashes / hex tokens).
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), REDACTED),
)

# base64 / base64url-like run; only redacted if it mixes upper, lower and digits.
_B64_CANDIDATE = re.compile(r"[A-Za-z0-9+/_\-]{40,}={0,2}")


def _b64_replace(match: re.Match[str]) -> str:
    s = match.group(0)
    if (
        any(c.isupper() for c in s)
        and any(c.islower() for c in s)
        and any(c.isdigit() for c in s)
    ):
        return REDACTED
    return s


def _secret_env_values(env_names: Iterable[str] = SECRET_ENV_VARS) -> list[str]:
    values = []
    for name in env_names:
        v = os.environ.get(name)
        if v and len(v.strip()) >= _MIN_ENV_VALUE_LEN:
            values.append(v.strip())
    # Longest first so that overlapping values are fully masked.
    return sorted(set(values), key=len, reverse=True)


def redact(text: object, env_names: Iterable[str] = SECRET_ENV_VARS) -> str:
    """Return `text` (stringified) with secret-looking substrings replaced by [REDACTED]."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    # 1) exact values from the environment (read at call time).
    for value in _secret_env_values(env_names):
        s = s.replace(value, REDACTED)
    # 2) generic patterns.
    for pattern, repl in _PATTERNS:
        s = pattern.sub(repl, s)
    s = _B64_CANDIDATE.sub(_b64_replace, s)
    return s


class RedactingFilter(logging.Filter):
    """logging.Filter that rewrites record.msg/args into a redacted, pre-formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed %-args: fall back to raw msg
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        return True


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts the final string (covers tracebacks / exc_info)."""

    def __init__(self, inner: logging.Formatter | None = None):
        super().__init__()
        self._inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


class RedactingStream(io.TextIOBase):
    """Text stream wrapper that redacts each write (for print() output such as banners).

    Limitation: a secret split across two separate write() calls may not be detected.
    """

    def __init__(self, inner):
        self._inner = inner

    def write(self, s: str) -> int:
        self._inner.write(redact(s))
        return len(s)

    def flush(self) -> None:
        self._inner.flush()

    def isatty(self) -> bool:
        try:
            return self._inner.isatty()
        except Exception:
            return False

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._inner, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._inner.fileno()

    def writable(self) -> bool:
        return True

    def __getattr__(self, name):
        # Delegate anything else (e.g. .buffer, .errors) to the wrapped stream.
        # Note: writes through .buffer bypass redaction.
        return getattr(self._inner, name)


def install_redaction(logger: logging.Logger | None = None) -> None:
    """Attach RedactingFilter + RedactingFormatter to every handler of `logger` (root by default).

    Filters on handlers (not on the logger) are needed so that records propagated from
    child loggers (forecasting_tools, litellm, httpx, ...) are also redacted.
    """
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
        if not isinstance(handler.formatter, RedactingFormatter):
            handler.setFormatter(RedactingFormatter(handler.formatter))
