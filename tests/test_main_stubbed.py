"""Offline tests of the patched main.py using STUB modules.

forecasting_tools / bot_helpers / dotenv are replaced with tiny fakes, so no real
library, key, API call or network access is involved. This checks only the patch's own
control flow (MiniBench flag, cost-cap wiring, variant LLM selection, V2 aggregation,
OpenRouter provider field on outbound request bodies); behaviour of the real
forecasting-tools / litellm libraries is 未確認. The litellm stub never sends anything: it only
records the JSON body an OpenAI-compatible client would POST (extra_body merged at top level).
"""

import asyncio
import contextlib
import io
import logging
import os
import runpy
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

CALLS = []  # tournament ids passed to forecast_on_tournament
BODIES = []  # request bodies the litellm stub would have sent (never sent)
REQUIRED = {"data_collection": "deny", "zdr": True}
GENERAL_LLM_MOD = "forecasting_tools.ai_models.general_llm"  # stub path (real path 未確認)


def _stub_body(model, kwargs):
    body = {"model": model, "messages": kwargs.get("messages", [])}
    for k, v in kwargs.items():
        if k not in ("messages", "extra_body", "model"):
            body[k] = v
    body.update(kwargs.get("extra_body") or {})
    return body


def _install_litellm_stub():
    ll = types.ModuleType("litellm")

    async def acompletion(model=None, **kwargs):
        BODIES.append(_stub_body(model, kwargs))
        return {"choices": [{"message": {"content": "ok"}}]}

    def completion(model=None, **kwargs):
        BODIES.append(_stub_body(model, kwargs))
        return {"choices": [{"message": {"content": "ok"}}]}

    ll.acompletion = acompletion
    ll.completion = completion
    sys.modules["litellm"] = ll
    return ll


def _install_stubs(with_cost_manager=True, with_litellm=True):
    BODIES.clear()
    if with_litellm:
        ll = _install_litellm_stub()
    else:
        sys.modules.pop("litellm", None)
        ll = None
    ft = types.ModuleType("forecasting_tools")
    ft.__path__ = []  # package-like
    sub = types.ModuleType(GENERAL_LLM_MOD)
    if ll is not None:
        sub.acompletion = ll.acompletion  # like "from litellm import acompletion"

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __class_getitem__(cls, item):  # allow annotations like ReasonedPrediction[float]
            return cls

    names = [
        "AskNewsSearcher", "BinaryQuestion", "MetaculusQuestion", "MultipleChoiceQuestion",
        "NumericDistribution", "NumericQuestion", "DateQuestion", "DatePercentile", "Percentile",
        "ConditionalQuestion", "ConditionalPrediction", "PredictionTypes", "PredictionAffirmed",
        "BinaryPrediction", "PredictedOptionList", "ReasonedPrediction", "SmartSearcher",
    ]
    for n in names:
        setattr(ft, n, type(n, (_Any,), {}))

    class GeneralLlm:
        def __init__(self, model, **kw):
            self.model = model
            self.kw = kw

        @staticmethod
        def to_model_name(x):
            return getattr(x, "model", x)

        async def invoke(self, prompt):
            # Like the real wrapper: calls litellm's acompletion through a module-level alias.
            await sys.modules[GENERAL_LLM_MOD].acompletion(
                model=self.model, messages=[{"role": "user", "content": prompt}], **self.kw
            )
            return "reasoning"

    class ForecastBot:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.skip_previously_forecasted_questions = kwargs.get(
                "skip_previously_forecasted_questions"
            )

        def get_llm(self, name, kind=None):
            return f"default-llm:{name}"

        async def _aggregate_predictions(self, predictions, question):
            return "parent-aggregate"

        async def forecast_on_tournament(self, tid, return_exceptions=False):
            CALLS.append(tid)
            return [f"report:{tid}"]

        def log_report_summary(self, reports):
            pass

    class MetaculusClient:
        CURRENT_AI_COMPETITION_ID = "AIB-STUB"
        CURRENT_MINIBENCH_ID = "MINIBENCH-STUB"
        CURRENT_METACULUS_CUP_ID = "CUP-STUB"

    class MonetaryCostManager:
        instances = []

        def __init__(self, max_cost=None):
            self.max_cost = max_cost
            self.current_usage = 0.0
            MonetaryCostManager.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ft.GeneralLlm = GeneralLlm
    ft.ForecastBot = ForecastBot
    ft.MetaculusClient = MetaculusClient
    ft.clean_indents = lambda s: s
    ft.structure_output = None
    if with_cost_manager:
        ft.MonetaryCostManager = MonetaryCostManager
    sys.modules["forecasting_tools"] = ft
    sys.modules[GENERAL_LLM_MOD] = sub

    bh = types.ModuleType("bot_helpers")
    bh.check_environment = lambda strict=True: None
    bh.print_run_summary_banner = lambda *a, **k: None
    bh.print_startup_banner = lambda *a, **k: None
    bh.silence_noisy_dependencies = lambda: None
    sys.modules["bot_helpers"] = bh

    de = types.ModuleType("dotenv")
    de.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = de
    sys.modules.pop("main", None)
    return ft


@contextlib.contextmanager
def _preserve_logging_and_stdio():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    out, err = sys.stdout, sys.stderr
    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in handlers:
            root.addHandler(h)
        root.setLevel(level)
        sys.stdout, sys.stderr = out, err


def _run_main(env, mode="tournament"):
    CALLS.clear()
    base = {"BOT_REDACT_STDIO": "false"}
    base.update(env)
    with mock.patch.dict(os.environ, base, clear=False), \
         mock.patch.object(sys, "argv", ["main.py", "--mode", mode]), \
         _preserve_logging_and_stdio():
        return runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")


class TestMainFlow(unittest.TestCase):
    def setUp(self):
        for k in ("ENABLE_MINIBENCH", "BOT_VARIANT", "BOT_MAX_COST_USD", "BOT_ALLOW_NO_COST_CAP"):
            os.environ.pop(k, None)

    def test_minibench_off_by_default(self):
        _install_stubs()
        _run_main({})
        self.assertEqual(CALLS, ["AIB-STUB"])

    def test_minibench_on_with_flag(self):
        _install_stubs()
        g = _run_main({"ENABLE_MINIBENCH": "true"})
        self.assertEqual(CALLS, ["AIB-STUB", "MINIBENCH-STUB"])
        self.assertEqual(g["forecast_reports"], ["report:AIB-STUB", "report:MINIBENCH-STUB"])

    def test_cost_manager_created_with_cap(self):
        ft = _install_stubs()
        _run_main({"BOT_MAX_COST_USD": "0.25"})
        self.assertEqual(ft.MonetaryCostManager.instances[-1].max_cost, 0.25)

    def test_minibench_skipped_when_cap_reached(self):
        ft = _install_stubs()
        orig = ft.ForecastBot.forecast_on_tournament

        async def spend(self, tid, return_exceptions=False):
            ft.MonetaryCostManager.instances[-1].current_usage = 5.0
            return await orig(self, tid, return_exceptions)

        ft.ForecastBot.forecast_on_tournament = spend
        _run_main({"ENABLE_MINIBENCH": "true", "BOT_MAX_COST_USD": "1"})
        self.assertEqual(CALLS, ["AIB-STUB"])

    def test_fail_closed_without_cost_manager(self):
        _install_stubs(with_cost_manager=False)
        with self.assertRaises(SystemExit):
            _run_main({})
        self.assertEqual(CALLS, [])

    def test_override_without_cost_manager(self):
        _install_stubs(with_cost_manager=False)
        _run_main({"BOT_ALLOW_NO_COST_CAP": "true"})
        self.assertEqual(CALLS, ["AIB-STUB"])

    def test_variant_kwargs(self):
        _install_stubs()
        g = _run_main({"BOT_VARIANT": "V3", "BOT_STRONG_MODEL": "strong-x", "BOT_CHEAP_MODEL": "cheap-x"})
        kw = g["template_bot"].init_kwargs
        self.assertEqual(kw["predictions_per_research_report"], 1)
        self.assertEqual(kw["llms"]["default"].model, "strong-x")
        self.assertEqual(kw["llms"]["researcher"].model, "cheap-x")
        self.assertTrue(kw["skip_previously_forecasted_questions"])

    def test_template_variant_passes_no_llms(self):
        _install_stubs()
        g = _run_main({"BOT_VARIANT": "TEMPLATE"})
        self.assertNotIn("llms", g["template_bot"].init_kwargs)
        self.assertEqual(g["template_bot"].init_kwargs["predictions_per_research_report"], 5)

    def test_v1_no_research(self):
        _install_stubs()
        g = _run_main({"BOT_VARIANT": "V1"})
        self.assertEqual(g["template_bot"].init_kwargs["llms"]["researcher"], "no_research")


class TestBotMethods(unittest.TestCase):
    def _bot(self, variant_name, env=None):
        _install_stubs()
        import importlib
        main = importlib.import_module("main")
        import bot_config
        e = {"BOT_STRONG_MODEL": "S", "BOT_CHEAP_MODEL": "C"}
        e.update(env or {})
        bot = main.FallTemplateBot2026()
        bot.configure_variant(bot_config.build_variant(variant_name, e))
        return bot

    def test_v4_split_per_question(self):
        bot = self._bot("V4")
        q1 = types.SimpleNamespace(page_url="u1")
        q2 = types.SimpleNamespace(page_url="u2")
        m1 = [bot._forecaster_llm(q1).model for _ in range(5)]
        m2 = [bot._forecaster_llm(q2).model for _ in range(5)]
        self.assertEqual(sorted(m1), ["C", "C", "C", "S", "S"])
        self.assertEqual(m1, m2)

    def test_template_uses_default_llm(self):
        bot = self._bot("TEMPLATE")
        self.assertEqual(bot._forecaster_llm(types.SimpleNamespace(page_url="u")), "default-llm:default")

    def test_v2_trimmed_mean_binary(self):
        bot = self._bot("V2")
        out = asyncio.run(bot._aggregate_predictions([0.1, 0.4, 0.5, 0.6, 0.99], None))
        self.assertAlmostEqual(out, 0.5)

    def test_v2_delegates_non_binary(self):
        bot = self._bot("V2")
        out = asyncio.run(bot._aggregate_predictions([object(), object()], None))
        self.assertEqual(out, "parent-aggregate")

    def test_v0_delegates(self):
        bot = self._bot("V0")
        out = asyncio.run(bot._aggregate_predictions([0.1, 0.2, 0.3, 0.4], None))
        self.assertEqual(out, "parent-aggregate")


class TestOpenRouterProvider(unittest.TestCase):
    """Every outbound OpenRouter request body must carry provider = {"data_collection": "deny", "zdr": true}."""

    def setUp(self):
        for k in ("ENABLE_MINIBENCH", "BOT_VARIANT", "BOT_MAX_COST_USD", "BOT_ALLOW_NO_COST_CAP",
                  "BOT_CHEAP_MODEL", "BOT_STRONG_MODEL", "BOT_PARSER_MODEL", "BOT_RESEARCHER"):
            os.environ.pop(k, None)

    def _assert_all_bodies_protected(self, min_count=1):
        self.assertGreaterEqual(len(BODIES), min_count)
        for body in BODIES:
            self.assertTrue(str(body["model"]).startswith("openrouter/"), body["model"])
            self.assertIn("provider", body)
            self.assertEqual(body["provider"]["data_collection"], "deny")
            self.assertIs(body["provider"]["zdr"], True)

    def test_guard_installed_by_main(self):
        _install_stubs()
        _run_main({})
        import bot_config
        ll = sys.modules["litellm"]
        self.assertTrue(getattr(ll.acompletion, bot_config._GUARD_ATTR, False))
        self.assertTrue(getattr(ll.completion, bot_config._GUARD_ATTR, False))
        self.assertIs(sys.modules[GENERAL_LLM_MOD].acompletion, ll.acompletion)

    def test_production_default_v0_all_llms(self):
        _install_stubs()
        g = _run_main({})  # production default: BOT_VARIANT unset -> V0, default OpenRouter models
        self.assertEqual(g["variant"].name, "V0")
        llms = g["template_bot"].init_kwargs["llms"]
        for role in ("default", "summarizer", "researcher", "parser"):
            self.assertTrue(hasattr(llms[role], "invoke"), role)
            self.assertEqual(llms[role].kw["extra_body"]["provider"], REQUIRED, role)
        BODIES.clear()
        for role in ("default", "summarizer", "researcher", "parser"):
            asyncio.run(llms[role].invoke("q"))
        for llm in g["template_bot"]._variant_forecasters:
            asyncio.run(llm.invoke("q"))
        self._assert_all_bodies_protected(min_count=9)
        for body in BODIES:
            self.assertEqual(body["provider"], REQUIRED)

    def test_every_variant_sends_provider(self):
        for name in ("V0", "V1", "V2", "V3", "V4", "V5"):
            with self.subTest(variant=name):
                _install_stubs()
                g = _run_main({"BOT_VARIANT": name})
                bot = g["template_bot"]
                BODIES.clear()
                for llm in list(bot.init_kwargs["llms"].values()) + list(bot._variant_forecasters):
                    if hasattr(llm, "invoke"):
                        asyncio.run(llm.invoke("q"))
                q = types.SimpleNamespace(page_url="u")
                for _ in range(bot._variant.predictions_per_research_report):
                    asyncio.run(bot._forecaster_llm(q).invoke("q"))
                self._assert_all_bodies_protected(min_count=4)

    def test_template_library_default_llm_goes_through_choke_point(self):
        ft = _install_stubs()
        _run_main({"BOT_VARIANT": "TEMPLATE"})
        BODIES.clear()
        # A GeneralLlm the library builds itself (no extra kwargs from our patch).
        asyncio.run(ft.GeneralLlm("openrouter/library-default").invoke("q"))
        self._assert_all_bodies_protected()
        self.assertEqual(BODIES[-1]["provider"], REQUIRED)

    def test_caller_cannot_weaken_and_other_keys_kept(self):
        _install_stubs()
        _run_main({})
        BODIES.clear()
        ll = sys.modules["litellm"]
        weak = {"provider": {"data_collection": "allow", "zdr": False, "order": ["p1"]}}
        asyncio.run(ll.acompletion(model="openrouter/a/b", messages=[], extra_body=weak))
        self.assertEqual(BODIES[-1]["provider"], {"data_collection": "deny", "zdr": True, "order": ["p1"]})
        self.assertEqual(weak["provider"]["zdr"], False)  # caller's dict not mutated

    def test_sync_positional_and_custom_provider_paths(self):
        _install_stubs()
        _run_main({})
        BODIES.clear()
        ll = sys.modules["litellm"]
        ll.completion("openrouter/a/b", messages=[])
        ll.completion(model="anthropic/x", custom_llm_provider="openrouter", messages=[])
        ll.completion(model="anthropic/x", api_base="https://openrouter.ai/api/v1", messages=[])
        self.assertEqual(len(BODIES), 3)
        for body in BODIES:
            self.assertEqual(body["provider"], REQUIRED)

    def test_non_openrouter_untouched(self):
        _install_stubs()
        _run_main({})
        BODIES.clear()
        asyncio.run(sys.modules["litellm"].acompletion(model="metaculus/gpt-4o", messages=[]))
        self.assertNotIn("provider", BODIES[-1])

    def test_fail_closed_without_litellm(self):
        _install_stubs(with_litellm=False)
        with self.assertRaises(SystemExit):
            _run_main({})
        self.assertEqual(CALLS, [])

    def test_guard_idempotent(self):
        _install_stubs()
        import bot_config
        bot_config.install_openrouter_provider_guard()
        first = sys.modules["litellm"].acompletion
        self.assertEqual(bot_config.install_openrouter_provider_guard(), 0)
        self.assertIs(sys.modules["litellm"].acompletion, first)
        BODIES.clear()
        asyncio.run(first(model="openrouter/a", messages=[]))
        self.assertEqual(BODIES[-1]["provider"], REQUIRED)


def tearDownModule():
    for n in ("forecasting_tools", GENERAL_LLM_MOD, "litellm", "bot_helpers", "dotenv", "main"):
        sys.modules.pop(n, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
