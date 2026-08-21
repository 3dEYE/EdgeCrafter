"""Shared ONNX precision-policy metadata and sensitive-operation contracts."""

from pathlib import Path
from typing import Dict


PRECISION_POLICY_METADATA_KEY = "edgecrafter.precision_policy"

# Operations that should stay FP32 when a lower-precision policy asks for a
# conservative high-precision fallback.  This list is shared by the calibrated
# FP16 dataflow and ModelOpt FP8 Q/DQ exporters; it is not an export policy by
# itself.
SENSITIVE_FP32_OP_TYPES = frozenset(
    {
        "ArgMax",
        "ArgMin",
        "BatchNormalization",
        "Clip",
        "CumSum",
        "Div",
        "Exp",
        "GridSample",
        "GroupNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "Log",
        "LogSoftmax",
        "LpNormalization",
        "MeanVarianceNormalization",
        "NonMaxSuppression",
        "Pow",
        "Reciprocal",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "RMSNormalization",
        "Sigmoid",
        "Softmax",
        "Sqrt",
        "TopK",
    }
)


def _metadata(model) -> Dict[str, str]:
    return {entry.key: entry.value for entry in model.metadata_props}


def read_precision_policy(model_path: Path) -> str:
    """Return the EdgeCrafter precision policy recorded in an ONNX model."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    return _metadata(model).get(PRECISION_POLICY_METADATA_KEY, "baseline")
