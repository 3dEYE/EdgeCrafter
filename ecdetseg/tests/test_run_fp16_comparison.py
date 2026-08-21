import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.deployment.run_fp16_comparison import (
    assess_gates,
    get_parser,
    summarize_inspector,
)


class RunFp16ComparisonTests(unittest.TestCase):
    def test_model_and_dataset_inputs_are_required(self):
        parser = get_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data", "dataset"])

        args = parser.parse_args(
            ["--data", "dataset", "--config", "model.yml", "--checkpoint", "weights.pth"]
        )
        self.assertEqual(args.data, "dataset")
        self.assertEqual(args.config, "model.yml")
        self.assertEqual(args.checkpoint, "weights.pth")

    def test_gates_use_map_ar100_latency_and_engine_size(self):
        baseline = [0.5, 0.7, 0.5, 0.1, 0.2, 0.3, 0.2, 0.4, 0.6, 0.1, 0.2, 0.3]
        candidate = [0.4995, 0.7, 0.5, 0.1, 0.2, 0.3, 0.2, 0.4, 0.5995, 0.1, 0.2, 0.3]
        result = assess_gates(
            baseline,
            candidate,
            10.0,
            10.4,
            1000,
            1090,
            max_map_drop=0.001,
            max_ar100_drop=0.001,
            max_latency_regression=0.05,
            max_engine_size_regression=0.10,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(check["passed"] for check in result["checks"].values()))

        failed = assess_gates(
            baseline,
            candidate,
            10.0,
            10.6,
            1000,
            1090,
            max_map_drop=0.001,
            max_ar100_drop=0.001,
            max_latency_regression=0.05,
            max_engine_size_regression=0.10,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["latency_regression"]["passed"])

    def test_inspector_summary_counts_tensor_edge_types(self):
        payload = {
            "Layers": [
                {
                    "Inputs": [{"Format/Datatype": "Half"}, {"Format/Datatype": "Int32"}],
                    "Outputs": [{"Format/Datatype": "Float"}],
                },
                {"Outputs": [{"Format/Datatype": "FP16"}, {"Format/Datatype": "Int64"}]},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inspector.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary = summarize_inspector(path)

        self.assertEqual(summary["layers"], 2)
        self.assertEqual(
            summary["tensor_edges"],
            {"float16": 2, "float32": 1, "int32": 1, "int64": 1},
        )


if __name__ == "__main__":
    unittest.main()
