"""Local unit tests for bot_config (no network, no keys)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bot_config as bc  # noqa: E402


class TestFlags(unittest.TestCase):
    def test_minibench_default_off(self):
        self.assertFalse(bc.minibench_enabled({}))
        self.assertFalse(bc.minibench_enabled({"ENABLE_MINIBENCH": ""}))
        self.assertFalse(bc.minibench_enabled({"ENABLE_MINIBENCH": "false"}))
        self.assertFalse(bc.minibench_enabled({"ENABLE_MINIBENCH": "maybe"}))

    def test_minibench_on(self):
        for v in ("true", "TRUE", "1", "yes", "on", " True "):
            self.assertTrue(bc.minibench_enabled({"ENABLE_MINIBENCH": v}), v)

    def test_publish_default_true(self):
        self.assertTrue(bc.publish_enabled({}))
        self.assertFalse(bc.publish_enabled({"BOT_PUBLISH": "false"}))

    def test_allow_no_cap_default_false(self):
        self.assertFalse(bc.allow_no_cost_cap({}))


class TestMaxCost(unittest.TestCase):
    def test_default(self):
        self.assertEqual(bc.get_max_cost_usd({}), 1.00)

    def test_valid(self):
        self.assertEqual(bc.get_max_cost_usd({"BOT_MAX_COST_USD": "0.5"}), 0.5)

    def test_invalid_falls_back(self):
        for v in ("abc", "0", "-1", "nan", "inf"):
            self.assertEqual(bc.get_max_cost_usd({"BOT_MAX_COST_USD": v}), 1.00, v)


class TestVariants(unittest.TestCase):
    ENV = {"BOT_CHEAP_MODEL": "cheap-m", "BOT_STRONG_MODEL": "strong-m"}

    def test_default_variant_is_v0(self):
        self.assertEqual(bc.get_variant_name({}), "V0")
        self.assertEqual(bc.get_variant_name({"BOT_VARIANT": "v3"}), "V3")

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            bc.get_variant_name({"BOT_VARIANT": "V9"})

    def test_plan_table(self):
        e = self.ENV
        v0 = bc.build_variant("V0", e)
        self.assertEqual(v0.forecaster_models, ("cheap-m",) * 5)
        self.assertEqual(bc.resolve_researcher(v0), "cheap-m")
        self.assertEqual(v0.aggregation, "library_default")
        v1 = bc.build_variant("V1", e)
        self.assertEqual(bc.resolve_researcher(v1), "no_research")
        v2 = bc.build_variant("V2", e)
        self.assertEqual(v2.aggregation, "trimmed_mean")
        v3 = bc.build_variant("V3", e)
        self.assertEqual(v3.forecaster_models, ("strong-m",))
        self.assertEqual(v3.predictions_per_research_report, 1)
        v4 = bc.build_variant("V4", e)
        self.assertEqual(v4.forecaster_models.count("strong-m"), 2)
        self.assertEqual(v4.forecaster_models.count("cheap-m"), 3)
        v5 = bc.build_variant("V5", e)
        self.assertEqual(v5.forecaster_models, ("strong-m",) * 5)
        t = bc.build_variant("TEMPLATE", e)
        self.assertFalse(t.uses_custom_llms)
        self.assertEqual(t.predictions_per_research_report, 5)

    def test_parser_and_researcher_override(self):
        e = dict(self.ENV, BOT_PARSER_MODEL="p-m", BOT_RESEARCHER="asknews/news-summaries")
        v = bc.build_variant("V4", e)
        self.assertEqual(v.parser_model, "p-m")
        self.assertEqual(bc.resolve_researcher(v), "asknews/news-summaries")
        # V1 always disables research regardless of BOT_RESEARCHER
        self.assertEqual(bc.resolve_researcher(bc.build_variant("V1", e)), "no_research")

    def test_defaults_are_plan_candidates(self):
        v = bc.build_variant("V4", {})
        self.assertIn("openrouter/anthropic/claude-sonnet-5.5", v.forecaster_models)
        self.assertIn("openrouter/anthropic/claude-haiku-5.5", v.forecaster_models)


class TestHelpers(unittest.TestCase):
    def test_round_robin_per_question(self):
        c = {}
        seq_a = [bc.pick_forecaster_index(c, "qA", 5) for _ in range(5)]
        seq_b = [bc.pick_forecaster_index(c, "qB", 5) for _ in range(5)]
        self.assertEqual(seq_a, [0, 1, 2, 3, 4])
        self.assertEqual(seq_b, [0, 1, 2, 3, 4])
        self.assertEqual(bc.pick_forecaster_index(c, "qA", 5), 0)

    def test_trimmed_mean(self):
        self.assertAlmostEqual(bc.trimmed_mean([0.1, 0.4, 0.5, 0.6, 0.99]), 0.5)
        self.assertAlmostEqual(bc.trimmed_mean([0.2, 0.4]), 0.3)
        with self.assertRaises(ValueError):
            bc.trimmed_mean([])


class TestOpenRouterProvider(unittest.TestCase):
    REQ = {"data_collection": "deny", "zdr": True}

    def test_required_values_exact(self):
        self.assertEqual(bc.REQUIRED_OPENROUTER_PROVIDER, self.REQ)

    def test_kwargs_for_openrouter_model(self):
        self.assertEqual(bc.openrouter_provider_kwargs("openrouter/anthropic/claude-haiku-5.5"),
                         {"extra_body": {"provider": self.REQ}})
        self.assertEqual(bc.openrouter_provider_kwargs("metaculus/gpt-4o"), {})
        self.assertEqual(bc.openrouter_provider_kwargs("asknews/news-summaries"), {})

    def test_default_production_models_are_openrouter(self):
        v = bc.build_variant(environ={})  # production default (V0)
        self.assertEqual(v.name, "V0")
        for m in set(v.forecaster_models) | {v.cheap_model, v.parser_model, bc.resolve_researcher(v)}:
            self.assertTrue(bc.is_openrouter_request(m), m)
        self.assertTrue(bc.is_openrouter_request(bc.DEFAULT_STRONG_MODEL))

    def test_enforce_forces_and_keeps_other_keys(self):
        src = {"model": "openrouter/x", "extra_body": {"provider": {"data_collection": "allow", "zdr": False, "order": ["a"]}, "k": 1}}
        out = bc.enforce_openrouter_provider(src)
        self.assertEqual(out["extra_body"]["provider"], {"data_collection": "deny", "zdr": True, "order": ["a"]})
        self.assertEqual(out["extra_body"]["k"], 1)
        self.assertEqual(src["extra_body"]["provider"]["zdr"], False)  # input not mutated

    def test_enforce_top_level_provider_and_detection(self):
        out = bc.enforce_openrouter_provider({"model": "openrouter/x", "provider": {"zdr": False}})
        self.assertEqual(out["provider"], self.REQ)
        self.assertEqual(out["extra_body"]["provider"], self.REQ)
        self.assertTrue(bc.is_openrouter_request("x", {"custom_llm_provider": "OpenRouter"}))
        self.assertTrue(bc.is_openrouter_request("x", {"api_base": "https://openrouter.ai/api/v1"}))
        self.assertFalse(bc.is_openrouter_request("anthropic/claude", {}))

    def test_non_openrouter_unchanged(self):
        src = {"model": "metaculus/gpt-4o", "temperature": 0}
        self.assertEqual(bc.enforce_openrouter_provider(src), src)

    def test_provider_ok(self):
        self.assertTrue(bc.provider_ok({"provider": dict(self.REQ)}))
        self.assertFalse(bc.provider_ok({"provider": {"data_collection": "deny", "zdr": "true"}}))
        self.assertFalse(bc.provider_ok({}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
