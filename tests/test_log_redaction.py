"""Local unit tests for log_redaction (no network, dummy fake strings only)."""

import io
import logging
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from log_redaction import (  # noqa: E402
    REDACTED,
    RedactingFilter,
    RedactingFormatter,
    RedactingStream,
    install_redaction,
    redact,
)

# All values below are FAKE dummies made up for tests.
FAKE_OR_KEY = "sk-or-v1-" + "0123456789abcdefFAKEFAKEdummy0000"
FAKE_SK_KEY = "sk-" + "FAKEdummyKEY1234567890abcd"
FAKE_ANT_KEY = "sk-ant-api03-" + "FAKE_dummy-key_000111222333"
FAKE_HEX = "deadbeef" * 5  # 40 hex chars
FAKE_B64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0123456789abcdXYZ"
FAKE_METACULUS = "dummy-metaculus-token-value-xyz"
FAKE_OR_ENV = "not-a-real-openrouter-key-42"


class TestRedactFunction(unittest.TestCase):
    def test_sk_keys(self):
        for key in (FAKE_OR_KEY, FAKE_SK_KEY, FAKE_ANT_KEY):
            out = redact(f"using key {key} now")
            self.assertNotIn(key, out)
            self.assertIn(REDACTED, out)
            self.assertTrue(out.startswith("using key "))

    def test_bearer_and_token_headers(self):
        out = redact("Authorization: Bearer abc.DEF-ghi_jkl~123")
        self.assertEqual(out, f"Authorization: Bearer {REDACTED}")
        out = redact("headers={'Authorization': 'Token 1a2b3c4d5e6f7a8b'}")
        self.assertNotIn("1a2b3c4d5e6f7a8b", out)
        self.assertIn(f"Token {REDACTED}", out)

    def test_key_value_pairs(self):
        cases = [
            "api_key=FAKEvalue123",
            'METACULUS_TOKEN="FAKEvalue123"',
            "client_secret: FAKEvalue123",
            '{"password": "FAKEvalue123"}',
            "x-api-key: FAKEvalue123",
        ]
        for c in cases:
            out = redact(c)
            self.assertNotIn("FAKEvalue123", out, c)
            self.assertIn(REDACTED, out, c)

    def test_long_hex(self):
        out = redact(f"hash {FAKE_HEX} end")
        self.assertEqual(out, f"hash {REDACTED} end")

    def test_long_base64(self):
        out = redact(f"blob={FAKE_B64}==")
        self.assertNotIn(FAKE_B64, out)

    def test_env_values_masked(self):
        env = {"METACULUS_TOKEN": FAKE_METACULUS, "OPENROUTER_API_KEY": FAKE_OR_ENV}
        with mock.patch.dict(os.environ, env, clear=False):
            out = redact(f"tok {FAKE_METACULUS} / key {FAKE_OR_ENV} !")
        self.assertNotIn(FAKE_METACULUS, out)
        self.assertNotIn(FAKE_OR_ENV, out)
        self.assertEqual(out.count(REDACTED), 2)

    def test_short_env_values_not_replaced(self):
        with mock.patch.dict(os.environ, {"EXA_API_KEY": "true"}, clear=False):
            self.assertEqual(redact("value is true"), "value is true")

    def test_no_false_positives_on_normal_text(self):
        samples = [
            "Forecasted URL https://www.metaculus.com/questions/12345/ with prediction: 0.37.",
            "Probability: 42%",
            "prompt_tokens: 1500, completion_tokens: 320, max_tokens=4000",
            "Percentile 10: 2026-11-02",
            "Run cost (MonetaryCostManager): $0.0123 / cap $1.00",
            "tournament/fall-futureeval-2026/ and bot-testing-area",
            "日本語のログ行もそのまま",
        ]
        for s in samples:
            self.assertEqual(redact(s), s, s)

    def test_none_and_non_str(self):
        self.assertEqual(redact(None), "")
        self.assertEqual(redact(123), "123")


class TestLoggingIntegration(unittest.TestCase):
    def _make_logger(self, name):
        stream = io.StringIO()
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = False
        lg.setLevel(logging.INFO)
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        lg.addHandler(h)
        install_redaction(lg)
        return lg, stream

    def test_filter_with_args(self):
        lg, stream = self._make_logger("t.args")
        lg.info("key=%s header=%s", FAKE_OR_KEY, "Bearer FAKEbearer12345")
        out = stream.getvalue()
        self.assertNotIn(FAKE_OR_KEY, out)
        self.assertNotIn("FAKEbearer12345", out)
        self.assertTrue(out.startswith("INFO "))

    def test_child_logger_propagation(self):
        parent, stream = self._make_logger("t.parent")
        child = logging.getLogger("t.parent.child")
        child.propagate = True
        child.warning(f"leaked {FAKE_SK_KEY}")
        self.assertNotIn(FAKE_SK_KEY, stream.getvalue())

    def test_exception_traceback_redacted(self):
        lg, stream = self._make_logger("t.exc")
        try:
            raise RuntimeError(f"failed with {FAKE_ANT_KEY}")
        except RuntimeError:
            lg.exception("boom")
        out = stream.getvalue()
        self.assertIn("RuntimeError", out)
        self.assertNotIn(FAKE_ANT_KEY, out)

    def test_install_is_idempotent(self):
        lg, _ = self._make_logger("t.idem")
        install_redaction(lg)
        h = lg.handlers[0]
        self.assertEqual(sum(isinstance(f, RedactingFilter) for f in h.filters), 1)
        self.assertIsInstance(h.formatter, RedactingFormatter)
        self.assertNotIsInstance(h.formatter._inner, RedactingFormatter)

    def test_stream_wrapper(self):
        inner = io.StringIO()
        s = RedactingStream(inner)
        print(f"banner {FAKE_OR_KEY}", file=s)
        self.assertNotIn(FAKE_OR_KEY, inner.getvalue())
        self.assertIn(REDACTED, inner.getvalue())

    def test_stream_wrapper_delegates_attributes(self):
        s = RedactingStream(sys.__stdout__)
        self.assertTrue(s.writable())
        self.assertTrue(hasattr(s, "buffer"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
