import unittest

import export_public_results as public


class PublicExportTests(unittest.TestCase):
    def test_continuous_export_is_strictly_allowlisted(self):
        payload = public.continuous_payload(
            [
                {
                    "attempt": 7,
                    "round": 2,
                    "variant": "uzu-m-spec",
                    "context_tokens": 1000,
                    "output_tokens": 512,
                    "completion_tokens": 512,
                    "decode_tps": 20.0,
                    "status": "ok",
                    "attempt_log": "machine-local/run.log",
                    "run_dir": "machine-local/results",
                    "detail": "host machine-local-address",
                    "text": "generated output",
                }
            ]
        )
        row = payload["rows"][0]
        self.assertEqual(row["attempt"], 7)
        self.assertNotIn("attempt_log", row)
        self.assertNotIn("run_dir", row)
        self.assertNotIn("detail", row)
        self.assertNotIn("text", row)

    def test_archived_and_unknown_series_are_excluded(self):
        payload = public.continuous_payload(
            [
                {"attempt": 1, "variant": "uzu-m-spec", "status": "archived"},
                {"attempt": 2, "variant": "private-experiment", "status": "ok"},
            ]
        )
        self.assertEqual(payload["rows"], [])


if __name__ == "__main__":
    unittest.main()
