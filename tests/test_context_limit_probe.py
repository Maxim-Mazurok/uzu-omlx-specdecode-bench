import importlib.util
import unittest
from pathlib import Path


SPECIFICATION = importlib.util.spec_from_file_location(
    "context_limit_probe", Path(__file__).parents[1] / "context_limit_probe.py"
)
assert SPECIFICATION is not None
context_limit_probe = importlib.util.module_from_spec(SPECIFICATION)
assert SPECIFICATION.loader is not None
SPECIFICATION.loader.exec_module(context_limit_probe)


def observation(context_tokens: int, status: str) -> dict:
    return {"context_tokens": context_tokens, "status": status}


class ContextLimitProbeTests(unittest.TestCase):
    def test_starts_at_required_context_then_checks_maximum_after_pass(self):
        self.assertEqual(context_limit_probe.next_context([], 256 * 1024, 128 * 1024, 32 * 1024, 8 * 1024), 128 * 1024)
        self.assertEqual(
            context_limit_probe.next_context(
                [observation(128 * 1024, "ok")],
                256 * 1024,
                128 * 1024,
                32 * 1024,
                8 * 1024,
            ),
            256 * 1024,
        )

    def test_bisects_between_128k_pass_and_256k_failure(self):
        observations = [
            observation(256 * 1024, "ram_limit"),
            observation(128 * 1024, "ok"),
        ]
        self.assertEqual(
            context_limit_probe.next_context(
                observations, 256 * 1024, 128 * 1024, 32 * 1024, 8 * 1024
            ),
            192 * 1024,
        )

    def test_descends_below_failed_requirement(self):
        observations = [
            observation(128 * 1024, "ram_limit"),
        ]
        self.assertEqual(
            context_limit_probe.next_context(
                observations, 256 * 1024, 128 * 1024, 32 * 1024, 8 * 1024
            ),
            64 * 1024,
        )

    def test_stops_at_requested_precision(self):
        observations = [
            observation(256 * 1024, "ram_limit"),
            observation(128 * 1024, "ok"),
            observation(136 * 1024, "ram_limit"),
        ]
        self.assertIsNone(
            context_limit_probe.next_context(
                observations, 256 * 1024, 128 * 1024, 32 * 1024, 8 * 1024
            )
        )


if __name__ == "__main__":
    unittest.main()