import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.deployment.onnx_dataflow_fp16 import (  # noqa: E402
    DATAFLOW_FP16_POLICY,
    DEFAULT_FP32_NODE_PATTERNS,
    apply_dataflow_fp16_precision,
)
from tools.deployment.onnx_precision import read_precision_policy  # noqa: E402


def _make_dynamic_model():
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT16, ["N", 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N", 4])
    weight = np.eye(4, dtype=np.float32)
    weight[0, 0] = 1.0e-9
    nodes = [
        helper.make_node("Cast", ["images"], ["images_float"], to=TensorProto.FLOAT),
        helper.make_node("MatMul", ["images_float", "weight"], ["scores"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dataflow-fp16-test",
        [images],
        [scores],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],
        ir_version=10,
    )
    checker.check_model(model)
    return model


class DataflowFp16OnnxTest(unittest.TestCase):
    def test_restores_dynamic_contract_and_records_real_calibration(self):
        captured = {}

        def fake_converter(**kwargs):
            captured.update(kwargs)
            converted = onnx.load(kwargs["onnx_path"], load_external_data=False)
            self.assertEqual(converted.graph.input[0].type.tensor_type.shape.dim[0].dim_value, 1)
            converted.graph.value_info.append(
                helper.make_tensor_value_info("images_float", TensorProto.FLOAT, [1, 4])
            )
            return converted

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.onnx"
            calibration_path = root / "calibration.npz"
            report_path = root / "report.json"
            onnx.save(_make_dynamic_model(), model_path)
            np.savez(calibration_path, images=np.array([[1.0, -2.0, 3.0, 4.0]], np.float16))

            report = apply_dataflow_fp16_precision(
                model_path,
                calibration_path,
                report_path,
                converter=fake_converter,
            )
            rewritten = onnx.load(model_path, load_external_data=False)
            checker.check_model(rewritten)

            input_dim = rewritten.graph.input[0].type.tensor_type.shape.dim[0]
            output_dim = rewritten.graph.output[0].type.tensor_type.shape.dim[0]
            self.assertEqual(input_dim.dim_param, "N")
            self.assertEqual(output_dim.dim_param, "N")
            self.assertEqual(len(rewritten.graph.value_info), 0)
            self.assertEqual(read_precision_policy(model_path), DATAFLOW_FP16_POLICY)
            self.assertEqual(captured["nodes_to_exclude"], list(DEFAULT_FP32_NODE_PATTERNS))
            np.testing.assert_array_equal(
                captured["calibration_data"]["images"],
                np.array([[1.0, -2.0, 3.0, 4.0]], np.float16),
            )

            self.assertEqual(report["calibration"]["batch_size"], 1)
            self.assertEqual(report["calibration"]["inputs"]["images"]["max_abs"], 4.0)
            self.assertEqual(report["source_float_weights"]["nonzero_below_fp16_subnormal_count"], 1)
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["policy"], DATAFLOW_FP16_POLICY)
            self.assertEqual(persisted["validation"]["onnx_checker"], "passed")

    def test_rejects_calibration_keys_that_do_not_match_model_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.onnx"
            calibration_path = root / "calibration.npz"
            onnx.save(_make_dynamic_model(), model_path)
            np.savez(calibration_path, wrong=np.ones((1, 4), np.float16))

            with self.assertRaisesRegex(ValueError, "exactly match ONNX inputs"):
                apply_dataflow_fp16_precision(
                    model_path,
                    calibration_path,
                    converter=lambda **_: _make_dynamic_model(),
                )


if __name__ == "__main__":
    unittest.main()
