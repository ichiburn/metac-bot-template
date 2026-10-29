"""
Run-time configuration switches (added by local patch, not part of the official template).

Everything here is pure standard library so it can be unit-tested without
forecasting-tools, API keys or network access.

Environment variables (all optional):
  ENABLE_MINIBENCH        "true"/"1"/"yes"/"on" to also forecast on MiniBench in --mode tournament.
                          Default: off.
  BOT_VARIANT             V0 | V1 | V2 | V3 | V4 | V5 | TEMPLATE.  Default: V0.
  BOT_MAX_COST_USD        Per-run spending cap in USD for MonetaryCostManager. Default: 1.00.
  BOT_ALLOW_NO_COST_CAP   "true" to run even if MonetaryCostManager cannot be imported. Default: off
                          (fail closed).
  BOT_CHEAP_MODEL         litellm model id for the cheap model.
  BOT_STRONG_MODEL        litellm model id for the high-performance model.
  BOT_PARSER_MODEL        litellm model id for structure_output parsing (default: cheap model).
  BOT_RESEARCHER          Researcher for V0,V2-V5 (default: "cheap" = cheap LLM without web search).
                          e.g. "asknews/news-summaries", "no_research", or any litellm model id.
  BOT_PUBLISH             "false" to not post to Metaculus (dry run). Default: true (template behaviour).

Data protection (not configurable by env): every request to OpenRouter carries
  "provider": {"data_collection": "deny", "zdr": true}
(dual protection together with the OpenRouter account privacy settings, owner checklist O6).
See openrouter_provider_kwargs() / enforce_openrouter_provider() below.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import os
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# PLAN.md §3.3 candidates (OpenRouter ids present in evidence/openrouter_models_20261019.json).
# Which candidate is used in production is NOT decided in PLAN.md (decided on 10/28) -> override via env.
DEFAULT_CHEAP_MODEL = "openrouter/anthropic/claude-haiku-5.5"
DEFAULT_STRONG_MODEL = "openrouter/anthropic/claude-sonnet-5.5"

DEFAULT_MAX_COST_USD = 1.00
DEFAULT_VARIANT = "V0"
VALID_VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5", "TEMPLATE")


def env_flag(name: str, default: bool = False, environ: dict | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    logger.warning(f"Unrecognised value for {name}; using default={default}")
    return default


def minibench_enabled(environ: dict | None = None) -> bool:
    """MiniBench participation flag. Default OFF (PLAN.md: first MiniBench = 2026-11-02 09:00 JST)."""
    return env_flag("ENABLE_MINIBENCH", default=False, environ=environ)


def publish_enabled(environ: dict | None = None) -> bool:
    return env_flag("BOT_PUBLISH", default=True, environ=environ)


def allow_no_cost_cap(environ: dict | None = None) -> bool:
    return env_flag("BOT_ALLOW_NO_COST_CAP", default=False, environ=environ)


def get_max_cost_usd(environ: dict | None = None) -> float:
    """Positive, finite float. Invalid / non-positive values fall back to the default (cap is never disabled)."""
    env = os.environ if environ is None else environ
    raw = env.get("BOT_MAX_COST_USD")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_COST_USD
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("BOT_MAX_COST_USD is not a number; using default")
        return DEFAULT_MAX_COST_USD
    if not math.isfinite(value) or value <= 0:
        logger.warning("BOT_MAX_COST_USD must be > 0; using default")
        return DEFAULT_MAX_COST_USD
    return value


@dataclass(frozen=True)
class VariantConfig:
    name: str
    # Model id per forecast call, in order. Length == predictions_per_research_report.
    # Empty tuple => TEMPLATE: do not pass llms= at all (forecasting-tools defaults).
    forecaster_models: tuple[str, ...]
    researcher: str  # "cheap" | "no_research" | other researcher string / model id
    aggregation: str  # "library_default" | "trimmed_mean"
    parser_model: str
    cheap_model: str
    description: str = ""
    research_reports_per_question: int = 1

    @property
    def predictions_per_research_report(self) -> int:
        return len(self.forecaster_models) if self.forecaster_models else 5

    @property
    def uses_custom_llms(self) -> bool:
        return bool(self.forecaster_models)


def get_variant_name(environ: dict | None = None) -> str:
    env = os.environ if environ is None else environ
    raw = (env.get("BOT_VARIANT") or "").strip().upper()
    if raw == "":
        return DEFAULT_VARIANT
    if raw not in VALID_VARIANTS:
        raise ValueError(
            f"BOT_VARIANT must be one of {', '.join(VALID_VARIANTS)} (got an unknown value)"
        )
    return raw


def build_variant(name: str | None = None, environ: dict | None = None) -> VariantConfig:
    """PLAN.md §3.2 variants V0-V5 (+ TEMPLATE = unmodified template model selection)."""
    env = os.environ if environ is None else environ
    name = (name or get_variant_name(env)).upper()
    cheap = (env.get("BOT_CHEAP_MODEL") or "").strip() or DEFAULT_CHEAP_MODEL
    strong = (env.get("BOT_STRONG_MODEL") or "").strip() or DEFAULT_STRONG_MODEL
    parser = (env.get("BOT_PARSER_MODEL") or "").strip() or cheap
    researcher = (env.get("BOT_RESEARCHER") or "").strip() or "cheap"

    table = {
        "TEMPLATE": VariantConfig(
            "TEMPLATE", (), "library_default", "library_default", parser, cheap,
            "テンプレートのまま（llms= を渡さない。forecasting-tools の既定モデル）",
        ),
        "V0": VariantConfig(
            "V0", (cheap,) * 5, researcher, "library_default", parser, cheap,
            "安いモデル×5回、まとめ方はライブラリ既定（PLAN上は中央値）",
        ),
        "V1": VariantConfig(
            "V1", (cheap,) * 5, "no_research", "library_default", parser, cheap,
            "V0 から検索を外す（漏れ確認用の基準）",
        ),
        "V2": VariantConfig(
            "V2", (cheap,) * 5, researcher, "trimmed_mean", parser, cheap,
            "安いモデル×5回、平均＋外れ値抑制（最大・最小を除いた平均。二択のみ）",
        ),
        "V3": VariantConfig(
            "V3", (strong,), researcher, "library_default", parser, cheap,
            "高性能モデル×1回",
        ),
        "V4": VariantConfig(
            "V4", (strong, strong, cheap, cheap, cheap), researcher, "library_default", parser, cheap,
            "高性能×2回＋安いモデル×3回",
        ),
        "V5": VariantConfig(
            "V5", (strong,) * 5, researcher, "library_default", parser, cheap,
            "高性能モデル×5回",
        ),
    }
    if name not in table:
        raise ValueError(f"Unknown variant: {name}")
    return table[name]


def resolve_researcher(variant: VariantConfig) -> str:
    """'cheap' -> the cheap model id; other strings ("no_research", "asknews/...", model ids) as-is."""
    if variant.researcher == "cheap":
        return variant.cheap_model
    return variant.researcher


def pick_forecaster_index(counters: dict, key: object, n: int) -> int:
    """Round-robin index per question key: 0,1,2,...,n-1,0,... (used to split V4 into 2 strong + 3 cheap)."""
    if n <= 0:
        raise ValueError("n must be positive")
    i = counters.get(key, 0)
    counters[key] = i + 1
    return i % n


def trimmed_mean(values: list[float]) -> float:
    """Mean after dropping one min and one max (when >= 4 values). Plain mean otherwise."""
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("empty")
    if len(vals) >= 4:
        vals = sorted(vals)[1:-1]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------------------------
# OpenRouter data protection (added 2026-10-23). Per-request provider preferences, on top of the
# account-level privacy settings (OWNER_CHECKLIST O6). Fixed values: there is NO env switch to
# weaken them, and caller-supplied values for these two keys are always overwritten.
# ---------------------------------------------------------------------------------------------

REQUIRED_OPENROUTER_PROVIDER = {"data_collection": "deny", "zdr": True}


def is_openrouter_request(model: object = None, kwargs: dict | None = None) -> bool:
    """True if a litellm-style call goes to OpenRouter.

    Detected by: model id prefix "openrouter/", custom_llm_provider == "openrouter",
    or an api_base / base_url pointing at openrouter.ai.
    """
    kw = kwargs or {}
    name = model if model is not None else kw.get("model")
    name = getattr(name, "model", name)  # GeneralLlm-like objects
    if isinstance(name, str) and name.strip().lower().startswith("openrouter/"):
        return True
    if str(kw.get("custom_llm_provider") or "").strip().lower() == "openrouter":
        return True
    for key in ("api_base", "base_url"):
        if "openrouter.ai" in str(kw.get(key) or "").lower():
            return True
    return False


def _merged_provider(existing: object) -> dict:
    prov = dict(existing) if isinstance(existing, dict) else {}
    prov.update(REQUIRED_OPENROUTER_PROVIDER)  # cannot be weakened by callers
    return prov


def enforce_openrouter_provider(kwargs: dict, model: object = None) -> dict:
    """Return a copy of litellm kwargs with extra_body.provider forced for OpenRouter calls.

    - extra_body is how litellm forwards extra JSON fields to OpenAI-compatible APIs (OpenRouter).
    - Other provider keys (e.g. "order") are kept; data_collection/zdr are overwritten.
    - If a top-level "provider" kwarg is present (litellm OpenRouter param), it is forced too.
    - Non-OpenRouter calls are returned unchanged (sending "provider" elsewhere could be rejected).
    """
    out = dict(kwargs)
    if not is_openrouter_request(model, out):
        return out
    extra = dict(out.get("extra_body") or {})
    extra["provider"] = _merged_provider(extra.get("provider"))
    out["extra_body"] = extra
    if "provider" in out:
        out["provider"] = _merged_provider(out.get("provider"))
    return out


def openrouter_provider_kwargs(model: str) -> dict:
    """Extra GeneralLlm(...) kwargs for a model id: {"extra_body": {"provider": {...}}} for OpenRouter."""
    if not is_openrouter_request(model):
        return {}
    return {"extra_body": {"provider": dict(REQUIRED_OPENROUTER_PROVIDER)}}


def provider_ok(body: dict) -> bool:
    """Check an outbound JSON request body (as sent to OpenRouter)."""
    p = (body or {}).get("provider")
    return isinstance(p, dict) and p.get("data_collection") == "deny" and p.get("zdr") is True


# --- single choke point: wrap litellm's completion entry points -------------------------------
# forecasting-tools' GeneralLlm is "a wrapper around litellm's acompletion" (forecasting-tools
# README). Wrapping litellm.completion / acompletion (and every already-imported alias of them in
# litellm.* / forecasting_tools.* modules) means OpenRouter calls made by the library itself
# (structure_output, SmartSearcher, TEMPLATE default models, ...) also carry the provider field.
# Which exact litellm function / module path forecasting-tools uses internally is 未確認.

_GUARD_ATTR = "__openrouter_provider_guard__"
_LITELLM_ENTRY_POINTS = ("completion", "acompletion", "responses", "aresponses")


def _wrap_entry_point(fn):
    if getattr(fn, _GUARD_ATTR, False):
        return fn
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            model = args[0] if args else kwargs.get("model")
            return await fn(*args, **enforce_openrouter_provider(kwargs, model))

    else:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            model = args[0] if args else kwargs.get("model")
            return fn(*args, **enforce_openrouter_provider(kwargs, model))

    setattr(wrapper, _GUARD_ATTR, True)
    return wrapper


def install_openrouter_provider_guard(modules: dict | None = None) -> int:
    """Patch litellm entry points so every OpenRouter request gets the fixed provider field.

    Returns the number of references replaced. Raises ImportError if litellm is unavailable
    (main.py then refuses to run = fail closed). Idempotent.
    """
    import importlib

    litellm = importlib.import_module("litellm")
    mods = sys.modules if modules is None else modules
    replaced = 0
    originals = {}
    for name in _LITELLM_ENTRY_POINTS:
        fn = getattr(litellm, name, None)
        if fn is None or not callable(fn):
            continue
        original = getattr(fn, "__wrapped__", fn) if getattr(fn, _GUARD_ATTR, False) else fn
        wrapped = _wrap_entry_point(fn)
        originals[id(original)] = wrapped
        originals[id(fn)] = wrapped
        if wrapped is not fn:
            setattr(litellm, name, wrapped)
            replaced += 1
    # Aliases created by "from litellm import acompletion" in already-imported modules.
    for mod_name, mod in list(mods.items()):
        if mod is None or not (
            mod_name == "litellm" or mod_name.startswith(("litellm.", "forecasting_tools"))
        ):
            continue
        for attr in _LITELLM_ENTRY_POINTS:
            try:
                current = getattr(mod, attr, None)
            except Exception:  # lazy modules may raise on attribute access
                continue
            if current is None or getattr(current, _GUARD_ATTR, False):
                continue
            target = originals.get(id(current))
            if target is not None:
                try:
                    setattr(mod, attr, target)
                    replaced += 1
                except Exception:
                    pass
    return replaced
