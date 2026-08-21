# ModelOpt 0.46 FP8 Autotune export

`export_modelopt_fp8.py` runs one target-specific deployment pipeline:

1. strictly loads the selected config and checkpoint;
2. exports ONNX and applies the calibrated explicit FP16 dataflow policy;
3. uses real validation images for ModelOpt calibration;
4. lets ModelOpt 0.46 Autotune benchmark FP8 Q/DQ schemes with TensorRT on the target GPU;
5. builds the selected strongly typed TensorRT engine;
6. optionally evaluates COCO mAP and AR and checks them against a comparable reference.

The config and checkpoint are required arguments. There is no hard-coded ECDet-M model or
`best.pth` in the exporter.

## Environment

Use an isolated environment because the ModelOpt ONNX extra pins its own ONNX Runtime and
CUDA-side dependencies:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r ecdetseg\requirements-modelopt-autotune.txt
```

The exporter deliberately requires `nvidia-modelopt` 0.46.x and fails early for another
minor version. TensorRT and PyTorch must see the same target GPU used for deployment.

## Validate inputs without exporting

```powershell
.\.venv\Scripts\python.exe ecdetseg\tools\deployment\export_modelopt_fp8.py `
  --data C:\path\to\dataset `
  --config C:\path\to\model.yml `
  --checkpoint C:\path\to\weights.pth `
  --input-size 512 512 `
  --dry-run
```

The dry run checks the dataset/model class count, validation split, ONNX input shape,
checkpoint path, ModelOpt version, and Autotune settings.

## Export, Autotune, TensorRT, and COCO evaluation

```powershell
.\.venv\Scripts\python.exe ecdetseg\tools\deployment\export_modelopt_fp8.py `
  --data C:\path\to\dataset `
  --config C:\path\to\model.yml `
  --checkpoint C:\path\to\weights.pth `
  --input-size 512 512 `
  --calibration-samples 128 `
  --autotune-mode quick `
  --gpu auto-fp8 `
  --evaluate `
  --quality-reference C:\path\to\fp16-comparison.json
```

Autotune presets are the ModelOpt 0.46 budgets exposed by this adapter:

- `quick`: 30 schemes per region, 10 warmup runs, 50 timing runs;
- `default`: 50 schemes per region, 50 warmup runs, 100 timing runs;
- `extensive`: 200 schemes per region, 50 warmup runs, 200 timing runs.

Each budget can be overridden independently. `--autotune-state-file`,
`--autotune-pattern-cache`, and `--autotune-timing-cache` support resuming or reusing an
expensive search. `--autotune-node-filter` narrows the target regions without replacing
Autotune's node selection.

## Development and production GPU selection

The [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/) list native
FP8 Tensor Core throughput, so its optimal FP8 regions can differ materially from the RTX
4090 result. The production card does not have to be an L4: it may be any local GPU whose
compute capability and installed TensorRT expose FP8. Autotune latency decisions are still
target-specific:

- use RTX 4090 for exporter/API smoke tests and generic ONNX ABI checks;
- run the final Autotune search, TensorRT engine build, and COCO quality gate on the GPU
  class that will actually serve the model;
- use the default `--gpu auto-fp8` to select a local card whose compute capability and
  TensorRT API support FP8, or pass an explicit CUDA index;
- optionally add `--expected-gpu-regex` when a deployment really is tied to a named SKU;
- do not reuse an Autotune state or TensorRT timing cache between different GPU profiles.
  The exporter writes and validates hardware sidecars for these resumable files.

`auto-fp8` verifies declared FP8 support; it does not assume that FP8 is faster. If several
eligible GPUs are visible, pass the desired CUDA index or limit `CUDA_VISIBLE_DEVICES`.
ModelOpt Autotune then makes the performance decision against FP16 on that card.

TensorRT engines are target builds too; do not ship an engine built for another GPU profile.
NVIDIA's [engine compatibility documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
also recommends device-specific builds for optimal tactics. If an L4-selected Q/DQ ONNX is
evaluated on another GPU, rebuild the engine there and treat the result as a numerical
quality check only, not an L4 latency measurement.

On RTX 4090, Autotune still compares every candidate against the calibrated FP16 dataflow
baseline. If no FP8 scheme is faster, the export is successful with:

- `result: NO_FP8_BENEFIT`;
- `selected_precision: fp16` for the TensorRT build;
- zero FP8 Q/DQ nodes in the selected ONNX.

This is a valid performance result, not a quantization failure. Use `--require-fp8-qdq`
only when the output contract strictly requires explicit FP8 `QuantizeLinear` and
`DequantizeLinear`; it turns the same outcome into an error.

Autotune optimizes latency, not detection quality. A final decision still requires the full
validation set with `--evaluate --quality-reference`. The exporter rejects a reference made
with a different checkpoint or input resolution, then gates both `mAP@[.50:.95]` and AR@100.

The output directory contains the FP16 source ONNX and precision report, calibration data,
Autotune state/models/logs, the selected ONNX, a strongly typed TensorRT engine when
requested, and `export_report.json` with timing, graph, ABI, and COCO evidence.
