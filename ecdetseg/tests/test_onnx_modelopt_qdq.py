import argparse
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.deployment.export_modelopt_fp8 import (  # noqa: E402
    MODEL_OPT_AUTOTUNE_PRESETS,
    assess_quality,
    bind_autotune_cache_to_gpu,
    get_parser,
    parse_gpu_selector,
    resolve_calibration_eps,
    resolve_autotune_settings,
    validate_quality_reference_contract,
    validate_expected_gpu,
)
from tools.deployment.onnx_modelopt_qdq import (  # noqa: E402
    _restore_external_value_info,
    apply_modelopt_fp8_qdq,
    require_modelopt_autotune_version,
)


def _make_conv_model(path: Path) -> None:
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT16, ["N", 32, 8, 8])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 32, 8, 8])
    cast = helper.make_node("Cast", ["images"], ["images_fp32"], name="/Cast", to=TensorProto.FLOAT)
    weights1 = np.linspace(-0.5, 0.5, 32 * 32, dtype=np.float32).reshape(32, 32, 1, 1)
    weights2 = np.linspace(-0.25, 0.25, 32 * 32, dtype=np.float32).reshape(32, 32, 1, 1)
    conv1 = helper.make_node("Conv", ["images_fp32", "weight1"], ["features"], name="/Conv1")
    relu = helper.make_node("Relu", ["features"], ["features_relu"], name="/Relu")
    conv2 = helper.make_node("Conv", ["features_relu", "weight2"], ["output"], name="/Conv2")
    graph = helper.make_graph(
        [cast, conv1, relu, conv2],
        "modelopt_fp8_smoke",
        [images],
        [output],
        [
            numpy_helper.from_array(weights1, "weight1"),
            numpy_helper.from_array(weights2, "weight2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    checker.check_model(model)
    onnx.save(model, path)


class ModelOptQdqTests(unittest.TestCase):
    def test_restores_external_shape_metadata_after_modelopt_autocast(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.onnx"
            _make_conv_model(source_path)
            source = onnx.load(source_path)
            converted = onnx.load(source_path)
            last_dim = converted.graph.output[0].type.tensor_type.shape.dim[-1]
            last_dim.ClearField("dim_value")
            last_dim.dim_param = "lost_by_shape_inference"

            _restore_external_value_info(source, converted)

            self.assertEqual(
                converted.graph.output[0].type.tensor_type.shape.dim[-1].dim_value,
                8,
            )

    def test_quality_gate_checks_map_and_ar100(self):
        result = assess_quality(
            {"map_50_95": 0.58, "ar_100": 0.71},
            {"map_50_95": 0.578, "ar_100": 0.706},
            max_map_drop=0.005,
            max_ar100_drop=0.005,
        )
        self.assertTrue(result["passed"])

        failed = assess_quality(
            {"map_50_95": 0.58, "ar_100": 0.71},
            {"map_50_95": 0.57, "ar_100": 0.706},
            max_map_drop=0.005,
            max_ar100_drop=0.005,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["map_50_95_drop"]["passed"])

    def test_quality_reference_rejects_different_input_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "weights.pth"
            checkpoint.write_bytes(b"checkpoint")
            reference = Path(temp_dir) / "comparison.json"
            reference.write_text(
                json.dumps(
                    {
                        "settings": {"input": "float16[N,3,512,512]"},
                        "baseline": {
                            "coco_bbox": {"map_50_95": 0.5, "ar_100": 0.6}
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input size differs"):
                validate_quality_reference_contract(
                    reference,
                    checkpoint=checkpoint,
                    image_hw=(640, 640),
                )

    def test_expected_gpu_rejects_profile_on_wrong_device(self):
        with self.assertRaisesRegex(RuntimeError, "wrong GPU"):
            validate_expected_gpu({"name": "NVIDIA GeForce RTX 4090"}, r"NVIDIA L4$")
        validate_expected_gpu({"name": "NVIDIA L4"}, r"NVIDIA L4$")

    def test_autotune_cache_is_bound_to_hardware_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "autotuner_state.yaml"
            fingerprint = {
                "name": "GPU A",
                "compute_capability": [8, 9],
                "total_memory_bytes": 24,
                "tensorrt_version": "10.14",
            }
            sidecar = bind_autotune_cache_to_gpu(state, fingerprint)
            self.assertTrue(sidecar.is_file())
            state.write_text("patterns: []\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different hardware profile"):
                bind_autotune_cache_to_gpu(state, {**fingerprint, "name": "GPU B"})

    def test_export_cli_requires_dataset_config_and_checkpoint(self):
        parser = get_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data", "dataset"])
        args = parser.parse_args(
            ["--data", "dataset", "--config", "model.yml", "--checkpoint", "weights.pth"]
        )
        self.assertEqual(args.autotune_mode, "quick")
        self.assertEqual(args.autotune_node_filter, [])
        self.assertFalse(args.require_fp8_qdq)
        self.assertEqual(args.gpu, "auto-fp8")
        self.assertEqual(
            resolve_autotune_settings(args),
            MODEL_OPT_AUTOTUNE_PRESETS["quick"],
        )
        self.assertFalse(args.onnx_only)

    def test_gpu_selector_accepts_auto_or_explicit_index(self):
        self.assertEqual(parse_gpu_selector("auto-fp8"), "auto-fp8")
        self.assertEqual(parse_gpu_selector("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_gpu_selector("none")

    def test_autotune_preset_can_be_overridden(self):
        args = get_parser().parse_args(
            [
                "--data", "dataset",
                "--config", "model.yml",
                "--checkpoint", "weights.pth",
                "--autotune-mode", "extensive",
                "--autotune-num-schemes-per-region", "3",
                "--autotune-warmup-runs", "0",
                "--autotune-timing-runs", "2",
            ]
        )
        self.assertEqual(
            resolve_autotune_settings(args),
            {"num_schemes_per_region": 3, "warmup_runs": 0, "timing_runs": 2},
        )

    @mock.patch("tools.deployment.onnx_modelopt_qdq.metadata.version", return_value="0.40.0")
    def test_rejects_modelopt_without_046_autotune_contract(self, _version):
        with self.assertRaisesRegex(RuntimeError, "requires nvidia-modelopt 0.46"):
            require_modelopt_autotune_version()

    def test_explicit_calibration_ep_validation(self):
        self.assertEqual(resolve_calibration_eps(["cuda", "cpu"], 2), ["cuda:2", "cpu"])
        with self.assertRaises(ValueError):
            resolve_calibration_eps(["auto", "cpu"], 0)

    def test_modelopt_creates_real_fp8_qdq_and_preserves_io(self):
        try:
            import modelopt.onnx.quantization  # noqa: F401
        except ImportError:
            self.skipTest("nvidia-modelopt is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.onnx"
            output = root / "quantized.onnx"
            calibration = root / "calibration.npz"
            report_path = root / "report.json"
            _make_conv_model(source)
            np.savez_compressed(
                calibration,
                images=np.linspace(0, 1, 2 * 32 * 8 * 8, dtype=np.float16).reshape(2, 32, 8, 8),
            )

            report = apply_modelopt_fp8_qdq(
                source,
                output,
                calibration,
                calibration_shapes="images:1x32x8x8",
                calibration_method="max",
                calibration_eps=["cpu"],
                high_precision_dtype="fp32",
                autotune=False,
                report_path=report_path,
                log_level="INFO",
            )

            model = onnx.load(output)
            checker.check_model(model)
            self.assertGreater(report["graph"]["quantize_linear_nodes"], 0)
            self.assertGreater(report["graph"]["dequantize_linear_nodes"], 0)
            self.assertGreater(report["graph"]["adjacent_qdq_pairs"], 0)
            self.assertEqual(report["graph"]["quantized_weight_compute_nodes"], {"Conv": 2})
            self.assertEqual(report["graph"]["qdq_adjacent_compute_nodes"], {"Conv": 2})
            self.assertGreater(report["graph"]["quantized_zero_point_types"]["FLOAT8E4M3FN"], 0)
            self.assertEqual(
                [node.op_type for node in model.graph.node].count("QuantizeLinear"),
                report["graph"]["quantize_linear_nodes"],
            )
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["policy"], "modelopt_fp8_qdq_autotune_v1")
            self.assertFalse(saved_report["autotune"]["enabled"])


if __name__ == "__main__":
    unittest.main()
