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
    def test_single_observation_stdev_is_zero(self):
        self.assertEqual(specbench.sample_stdev([15.0]), 0.0)

    def test_synthetic_prompt_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "synthetic",
                        "expected_prompt_tokens": 10,
                        "synthetic_context": {
                            "system": "system",
                            "unit": "abc",
                            "repeats": 2,
                            "pad_unit": "x",
                            "pad_repeats": 1,
                            "suffix": "end",
                        },
                    }
                )
            )
            row = specbench.load_prompts(path)[0]
            self.assertEqual(row["messages"][1]["content"], "abcabcxend")
            self.assertNotIn("synthetic_context", row)

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

    def test_vlm_mtp_log(self):
        row = specbench.parse_spec_log(
            "vlm_mtp stats: request=abc finish=length rounds=35 "
            "accepted=29/35 (82.9%) tokens_per_round=1.83 "
            "emitted=64 block_size=2"
        )
        self.assertEqual(row["spec_acceptance_percent"], 82.9)
        self.assertEqual(row["spec_cycles"], 35)
        self.assertEqual(row["tokens_per_spec_cycle"], 1.83)
        self.assertEqual(row["spec_block_size"], 2)

    def test_vlm_mtp_fallback_is_rejected(self):
        variant = {"id": "vlm", "speculative": "vlm-mtp"}
        with self.assertRaisesRegex(RuntimeError, "fell back to LLM"):
            specbench.validate_speculative_runtime(
                variant, "VLM loading failed; falling back to LLM"
            )

    def test_vlm_mtp_requires_decode_evidence(self):
        variant = {"id": "vlm", "speculative": "vlm-mtp"}
        with self.assertRaisesRegex(RuntimeError, "without VLM-MTP decode evidence"):
            specbench.validate_speculative_runtime(variant, "ordinary completion")

    def test_disabled_variant_is_rejected(self):
        config = {"variants": [{"id": "off", "enabled": False}]}
        with self.assertRaisesRegex(ValueError, "disabled variants"):
            specbench.selected_variants(config, "off")

    def test_opt_in_variant_is_selectable_but_not_in_default_run(self):
        config = {
            "variants": [
                {"id": "normal"},
                {"id": "optional", "default_run": False},
            ]
        }
        self.assertEqual(
            [row["id"] for row in specbench.selected_variants(config, None)],
            ["normal"],
        )
        self.assertEqual(
            [row["id"] for row in specbench.selected_variants(config, "optional")],
            ["optional"],
        )

    def test_vlm_mtp_settings_use_external_drafter_only(self):
        settings = specbench.omlx_settings(
            {
                "model": "target",
                "speculative": "vlm-mtp",
                "vlm_mtp_draft_block_size": 2,
            },
            Path("/draft/dflash"),
            Path("/draft/qwen36-mtp"),
        )["models"]["target"]
        self.assertTrue(settings["vlm_mtp_enabled"])
        self.assertEqual(settings["vlm_mtp_draft_model"], "/draft/qwen36-mtp")
        self.assertEqual(settings["vlm_mtp_draft_block_size"], 2)
        self.assertFalse(settings["dflash_enabled"])
        self.assertFalse(settings["mtp_enabled"])

    def test_turboquant_settings_are_explicitly_opt_in(self):
        disabled = specbench.omlx_settings(
            {"model": "target", "speculative": "off"},
            Path("/draft/dflash"),
            Path("/draft/qwen36-mtp"),
        )["models"]["target"]
        enabled = specbench.omlx_settings(
            {
                "model": "target",
                "speculative": "off",
                "turboquant_kv_enabled": True,
                "turboquant_kv_bits": 4,
                "turboquant_skip_last": True,
            },
            Path("/draft/dflash"),
            Path("/draft/qwen36-mtp"),
        )["models"]["target"]

        self.assertFalse(disabled["turboquant_kv_enabled"])
        self.assertTrue(enabled["turboquant_kv_enabled"])
        self.assertEqual(enabled["turboquant_kv_bits"], 4.0)
        self.assertTrue(enabled["turboquant_skip_last"])

    def test_vlm_mtp_with_turboquant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            specbench.omlx_settings(
                {
                    "model": "target",
                    "speculative": "vlm-mtp",
                    "turboquant_kv_enabled": True,
                },
                Path("/draft/dflash"),
                Path("/draft/qwen36-mtp"),
            )

    def test_rejected_omlx_settings_are_detected_in_startup_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "server.log"
            log_path.write_text(
                "Failed to load settings for model 'target': incompatible settings\n"
            )
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                specbench.validate_omlx_startup_log(log_path)

    def test_memory_limits_reject_low_free_ram(self):
        with self.assertRaisesRegex(RuntimeError, "only 5% memory free"):
            specbench.enforce_memory_limits(
                {
                    "system_memory_free_percent": 5,
                    "system_swap_used_bytes": 0,
                },
                {"abort_memory_free_percent": 12, "max_swap_growth_gib": 4},
                0,
            )

    def test_memory_limits_reject_swap_growth(self):
        with self.assertRaisesRegex(RuntimeError, "swap grew"):
            specbench.enforce_memory_limits(
                {
                    "system_memory_free_percent": 50,
                    "system_swap_used_bytes": 5 * 1024**3,
                },
                {"abort_memory_free_percent": 12, "max_swap_growth_gib": 4},
                0,
            )

    def test_mtp_view_patches_config_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "org" / "model"
            source.mkdir(parents=True)
            (source / "config.json").write_text(
                json.dumps({"text_config": {"model_type": "qwen3_5_text"}})
            )
            tensors = {
                f"mtp.{name}": {
                    "dtype": "BF16",
                    "shape": [1],
                    "data_offsets": [index * 2, index * 2 + 2],
                }
                for index, name in enumerate(
                    [
                        "layers.0.input_layernorm.weight",
                        "layers.0.post_attention_layernorm.weight",
                        "layers.0.self_attn.q_norm.weight",
                        "layers.0.self_attn.k_norm.weight",
                        "pre_fc_norm_hidden.weight",
                        "pre_fc_norm_embedding.weight",
                        "norm.weight",
                    ]
                )
            }
            tensors["mtp.fc.weight"] = {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [14, 14],
            }
            header = json.dumps(tensors).encode()
            (source / "mtp.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + struct.pack("<7H", *([0] * 7))
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
            self.assertNotEqual(
                (view / "model" / "mtp.safetensors").stat().st_ino,
                (source / "mtp.safetensors").stat().st_ino,
            )
            with (view / "model" / "mtp.safetensors").open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                handle.seek(8 + size)
                self.assertEqual(struct.unpack("<7H", handle.read(14)), (0x3F80,) * 7)
            index = json.loads(
                (view / "model" / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(index["weight_map"]["mtp.fc.weight"], "mtp.safetensors")

    def test_external_vlm_mtp_view_excludes_native_mtp_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "org" / "model"
            source.mkdir(parents=True)
            (source / "config.json").write_text("{}")
            (source / "model.safetensors").write_bytes(b"target")
            (source / "mtp.safetensors").write_bytes(b"native sidecar")

            view = specbench.omlx_model_view(
                root / "models",
                {"model": "model", "speculative": "vlm-mtp"},
                root / "view",
            )

            self.assertTrue((view / "model" / "model.safetensors").is_file())
            self.assertFalse((view / "model" / "mtp.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
