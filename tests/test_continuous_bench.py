import importlib.util
import argparse
import json
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
            manifest = {
                "schema_version": 2,
                "variants": list(continuous.VARIANTS),
                "segments": [
                    {
                        "start_attempt": 0,
                        "start_round": 0,
                        "seed": 42,
                        "min_context": 69,
                        "max_context": 50000,
                        "output_tokens": 512,
                        "delay_seconds": 60.0,
                    }
                ],
            }
            continuous.render_chart(chart, events, manifest)
            text = chart.read_text()
            self.assertIn("20.00 tok/s", text)
            self.assertIn("RAM stop", text)
            self.assertIn("refresh", text)

    def test_changed_range_becomes_new_segment_after_partial_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            old = {
                "schema_version": 1,
                "seed": 20260908,
                "min_context": 69,
                "max_context": 50000,
                "output_tokens": 512,
                "delay_seconds": 60.0,
                "variants": list(continuous.VARIANTS),
            }
            (campaign / "campaign.json").write_text(json.dumps(old))
            events = [
                {
                    "attempt": index,
                    "round": index // 3,
                    "variant": continuous.VARIANTS[index % 3],
                }
                for index in range(95)
            ]
            ledger = "".join(json.dumps(row) + "\n" for row in events)
            (campaign / "events.jsonl").write_text(ledger)
            args = argparse.Namespace(
                campaign_dir=str(campaign),
                seed=20260908,
                min_context=69,
                max_context=13000,
                output_tokens=512,
                delay_seconds=60.0,
            )
            _, loaded, manifest = continuous.prepare_campaign(args)
            self.assertEqual((campaign / "events.jsonl").read_text(), ledger)
            self.assertEqual(loaded, events)
            self.assertEqual(len(manifest["segments"]), 2)
            self.assertEqual(manifest["segments"][1]["start_attempt"], 96)
            self.assertEqual(manifest["segments"][1]["start_round"], 32)
            self.assertEqual(
                continuous.segment_for_round(manifest, 31)[1]["max_context"],
                50000,
            )
            self.assertEqual(
                continuous.segment_for_round(manifest, 32)[1]["max_context"],
                13000,
            )

            _, _, prepared_again = continuous.prepare_campaign(args)
            self.assertEqual(len(prepared_again["segments"]), 2)

    def test_chart_keeps_old_range_when_new_segment_is_narrower(self):
        manifest = {
            "schema_version": 2,
            "variants": list(continuous.VARIANTS),
            "segments": [
                {
                    "start_attempt": 0,
                    "start_round": 0,
                    "seed": 1,
                    "min_context": 69,
                    "max_context": 50000,
                    "output_tokens": 512,
                    "delay_seconds": 0,
                },
                {
                    "start_attempt": 3,
                    "start_round": 1,
                    "seed": 1,
                    "min_context": 69,
                    "max_context": 13000,
                    "output_tokens": 512,
                    "delay_seconds": 0,
                },
            ],
        }
        events = [
            {
                "attempt": index,
                "round": 0,
                "variant": variant,
                "context_tokens": 49000,
                "decode_tps": 10.0 + index,
                "ttft_seconds": 1.0,
                "status": "ok",
            }
            for index, variant in enumerate(continuous.VARIANTS)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            chart = Path(temporary) / "chart.html"
            continuous.render_chart(chart, events, manifest)
            text = chart.read_text()
            self.assertIn("collecting 69–13,000 context", text)
            self.assertIn(">50k</text>", text)


if __name__ == "__main__":
    unittest.main()
