import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.deployment.export_trt_eval import _validate_engine_contract  # noqa: E402


def _onnx_contract():
    return {
        "inputs": {"images": "float16"},
        "outputs": {
            "labels": "int32",
            "boxes": "float32",
            "scores": "float32",
        },
        "input_shapes": {"images": (-1, 3, 512, 512)},
        "output_shapes": {
            "labels": (-1, -1),
            "boxes": (-1, -1, 4),
            "scores": (-1, -1),
        },
    }


def _engine_contract(image_shape, profiles):
    batch = image_shape[0]
    return {
        "inputs": {"images": "float16"},
        "outputs": {
            "labels": "int32",
            "boxes": "float32",
            "scores": "float32",
        },
        "input_shapes": {"images": image_shape},
        "output_shapes": {
            "labels": (batch, 300),
            "boxes": (batch, 300, 4),
            "scores": (batch, 300),
        },
        "profiles": profiles,
        "profile_count": 1,
    }


class ExportTrtContractTests(unittest.TestCase):
    def test_accepts_dynamic_onnx_collapsed_by_degenerate_profile(self):
        _validate_engine_contract(
            _engine_contract((1, 3, 512, 512), profiles={}),
            _onnx_contract(),
            image_hw=(512, 512),
            static_batch=False,
            min_batch=1,
            opt_batch=1,
            max_batch=1,
        )

    def test_rejects_fixed_engine_for_non_degenerate_dynamic_profile(self):
        with self.assertRaisesRegex(ValueError, "optimization profile mismatch"):
            _validate_engine_contract(
                _engine_contract((1, 3, 512, 512), profiles={}),
                _onnx_contract(),
                image_hw=(512, 512),
                static_batch=False,
                min_batch=1,
                opt_batch=2,
                max_batch=4,
            )

    def test_rejects_collapsed_engine_with_wrong_fixed_batch(self):
        with self.assertRaisesRegex(ValueError, "optimization profile mismatch"):
            _validate_engine_contract(
                _engine_contract((2, 3, 512, 512), profiles={}),
                _onnx_contract(),
                image_hw=(512, 512),
                static_batch=False,
                min_batch=1,
                opt_batch=1,
                max_batch=1,
            )

    def test_accepts_regular_dynamic_engine_with_profile(self):
        profile = {
            "images": (
                (1, 3, 512, 512),
                (2, 3, 512, 512),
                (4, 3, 512, 512),
            )
        }
        _validate_engine_contract(
            _engine_contract((-1, 3, 512, 512), profiles=profile),
            _onnx_contract(),
            image_hw=(512, 512),
            static_batch=False,
            min_batch=1,
            opt_batch=2,
            max_batch=4,
        )


if __name__ == "__main__":
    unittest.main()
