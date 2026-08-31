import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.deployment.export_fp16_dataflow import get_parser  # noqa: E402
from tools.deployment.export_trt_eval import export_onnx  # noqa: E402
from tools.deployment.onnx_dataflow_fp16 import DATAFLOW_FP16_POLICY  # noqa: E402


class ExportFp16DataflowTests(unittest.TestCase):
    def test_cli_requires_dataset_config_and_checkpoint(self):
        parser = get_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data", "dataset"])

        args = parser.parse_args(
            ["--data", "dataset", "--config", "model.yml", "--checkpoint", "weights.pth"]
        )
        self.assertEqual(args.calibration_samples, 1)
        self.assertEqual(args.calibration_batch_size, 1)
        self.assertEqual(args.gpu, 0)
        self.assertFalse(args.onnx_only)
        self.assertFalse(hasattr(args, "onnx_precision_policy"))
        self.assertEqual(DATAFLOW_FP16_POLICY, "explicit_fp16_dataflow_calibrated_v1")

    def test_conservative_fp16_policy_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "Unsupported ONNX precision policy"):
                export_onnx(
                    cfg=None,
                    checkpoint="weights.pth",
                    output_file=Path(temp_dir) / "model.onnx",
                    batch_size=1,
                    opset=20,
                    static_batch=False,
                    check=False,
                    simplify=False,
                    strict_load=True,
                    export_mode="normalized",
                    onnx_precision_policy="explicit-fp16",
                )


if __name__ == "__main__":
    unittest.main()
