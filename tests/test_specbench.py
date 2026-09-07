import importlib.util
import json
import tempfile
import unittest
import struct
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "specbench", Path(__file__).parents[1] / "specbench.py"
)
specbench = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(specbench)


class MetricsTests(unittest.TestCase):
    def test_mtp_log(self):
        row = specbench.parse_spec_log(
            "MTP[abc] finish=length tokens=64 cycles=30 tok/cycle=2.13 "
            "accept=42/60 (70.0%) emits[init=1,draft=42,bonus=0,verify=21]"
        )
        self.assertEqual(row["spec_acceptance_percent"], 70.0)
        self.assertEqual(row["spec_cycles"], 30)
        self.assertEqual(row["spec_accepted_tokens"], 42)

    def test_dflash_log(self):
        row = specbench.parse_spec_log(
            "DFlash generation complete: 64 tokens, 25.2 tok/s, "
            "acceptance=81.5%, cycles=7"
        )
        self.assertEqual(row["spec_acceptance_percent"], 81.5)
        self.assertAlmostEqual(row["tokens_per_spec_cycle"], 64 / 7)

    def test_disabled_variant_is_rejected(self):
        config = {"variants": [{"id": "off", "enabled": False}]}
        with self.assertRaisesRegex(ValueError, "disabled variants"):
            specbench.selected_variants(config, "off")

    def test_mtp_view_patches_config_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "org" / "model"
            source.mkdir(parents=True)
            (source / "config.json").write_text(
                json.dumps({"text_config": {"model_type": "qwen3_5_text"}})
            )
            header = json.dumps({"mtp.fc.weight": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}).encode()
            (source / "mtp.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.weight": "model.safetensors"}})
            )
            view = specbench.omlx_model_view(
                root / "models",
                {"model": "model", "speculative": "mtp"},
                root / "view",
            )
            patched = json.loads((view / "model" / "config.json").read_text())
            original = json.loads((source / "config.json").read_text())
            self.assertEqual(patched["text_config"]["mtp_num_hidden_layers"], 1)
            self.assertEqual(
                patched["quantization"]["language_model.mtp.layers.0.self_attn.o_proj"],
                {"bits": 4, "group_size": 64},
            )
            self.assertNotIn("mtp_num_hidden_layers", original["text_config"])
            self.assertTrue((view / "model" / "mtp.safetensors").is_file())
            index = json.loads(
                (view / "model" / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(index["weight_map"]["mtp.fc.weight"], "mtp.safetensors")


if __name__ == "__main__":
    unittest.main()
