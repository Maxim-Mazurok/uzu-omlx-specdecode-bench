import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "continuous_bench", Path(__file__).parents[1] / "continuous_bench.py"
)
continuous = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(continuous)


class ContinuousBenchTests(unittest.TestCase):
    def test_context_schedule_is_deterministic(self):
        first = [continuous.context_for_round(42, i, 69, 50000) for i in range(8)]
        second = [continuous.context_for_round(42, i, 69, 50000) for i in range(8)]
        self.assertEqual(first, second)
        self.assertTrue(all(69 <= value <= 50000 for value in first))

    def test_each_round_rotates_all_variants(self):
        for round_index in range(5):
            scheduled = {
                continuous.variant_for(round_index, slot)
                for slot in range(len(continuous.VARIANTS))
            }
            self.assertEqual(scheduled, set(continuous.VARIANTS))

    def test_prompt_token_arithmetic(self):
        for target in (69, 70, 1000, 49999, 50000):
            prompt = continuous.exact_prompt(target, 1)
            synthetic = prompt["synthetic_context"]
            calculated = (
                continuous.CHAT_OVERHEAD_AND_SUFFIX_TOKENS
                + continuous.UNIT_TOKENS * synthetic["repeats"]
                + synthetic["pad_repeats"]
            )
            self.assertEqual(calculated, target)

    def test_ram_stop_classification(self):
        status, detail = continuous.classify(
            1, "RuntimeError: RAM safety stop: swap grew 4.2 GiB"
        )
        self.assertEqual(status, "ram_limit")
        self.assertEqual(detail, "swap grew 4.2 GiB")

    def test_chart_contains_success_and_ram_stop(self):
        events = [
            {
                "attempt": 0,
                "round": 0,
                "variant": "uzu-m-spec",
                "context_tokens": 1000,
                "decode_tps": 20.0,
                "ttft_seconds": 5.0,
                "status": "ok",
            },
            {
                "attempt": 1,
                "round": 0,
                "variant": "omlx-optiq-dflash",
                "context_tokens": 1000,
                "status": "ram_limit",
                "detail": "swap grew",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            chart = Path(temporary) / "chart.html"
            continuous.render_chart(chart, events, 42, 50000, 512)
            text = chart.read_text()
            self.assertIn("20.00 tok/s", text)
            self.assertIn("RAM stop", text)
            self.assertIn("refresh", text)


if __name__ == "__main__":
    unittest.main()
