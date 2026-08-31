import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "deployment" / "select_calibration_subset.py"
SPEC = importlib.util.spec_from_file_location("select_calibration_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _args(samples: int) -> argparse.Namespace:
    return argparse.Namespace(
        samples=samples,
        min_per_class=2,
        min_per_brightness=1,
        min_per_scale=1,
        min_cctv_high=2,
        min_cctv_low=2,
        dedup_cosine_threshold=0.99999,
    )


def test_select_subset_is_exact_deterministic_and_covers_constraints(tmp_path):
    rng = np.random.default_rng(42)
    samples = []
    stats = []
    for index in range(60):
        class_id = index % 3
        relative = Path(f"frame_{index:03d}.jpg")
        samples.append(
            MODULE.Sample(
                image=tmp_path / relative,
                label=tmp_path / relative.with_suffix(".txt"),
                relative_image=relative,
                classes=(class_id,),
                object_classes=(class_id,),
                median_box_area=(0.005, 0.05, 0.2)[index % 3],
            )
        )
        stats.append(
            MODULE.ImageStats(
                brightness=(index % 5 + 0.5) / 5.0,
                contrast=0.1 + index / 1000.0,
                saturation=0.2,
                sharpness=0.01 + index / 10000.0,
                width=640,
                height=480,
                grayscale=index == 0,
            )
        )
    dino = rng.normal(size=(60, 16)).astype(np.float32)
    siglip = rng.normal(size=(60, 12)).astype(np.float32)
    clusters = np.asarray([index % 5 for index in range(60)], dtype=np.int32)
    cctv = np.linspace(0.0, 1.0, 60, dtype=np.float32)

    first = MODULE.select_subset(samples, stats, dino, siglip, clusters, cctv, _args(20))
    second = MODULE.select_subset(samples, stats, dino, siglip, clusters, cctv, _args(20))

    assert first.selected == second.selected
    assert len(first.selected) == 20
    assert len(set(first.selected)) == 20
    assert {clusters[index] for index in first.selected} == set(range(5))
    for class_id in range(3):
        assert sum(class_id in samples[index].classes for index in first.selected) >= 2


def test_materialize_subset_preserves_pairs_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    samples = []
    report_entries = []
    for index in range(3):
        relative = Path("nested") / f"frame_{index}.jpg"
        image = source / "images" / "val" / relative
        label = source / "labels" / "val" / relative.with_suffix(".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color=(index, index, index)).save(image)
        label.write_text(f"0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        samples.append(
            MODULE.Sample(image, label, relative, (0,), (0,), 0.04)
        )
        report_entries.append({"relative_image": relative.as_posix()})

    output = tmp_path / "subset"
    report = {"samples": report_entries}
    data_cfg = {"nc": 1, "names": {0: "person"}}
    MODULE.materialize_subset(output, samples, report, data_cfg, "copy")

    assert MODULE.validate_materialized(output, 3) == {
        "images": 3,
        "labels": 3,
        "missing": 0,
        "orphan": 0,
    }
    assert "val: images/val" in (output / "data.yaml").read_text(encoding="utf-8")
    try:
        MODULE.materialize_subset(output, samples, report, data_cfg, "copy")
    except MODULE.SelectionError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("existing output must not be overwritten")


def test_resolve_yolo_validation_ignores_stale_path(tmp_path):
    root = tmp_path / "dataset"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "path: C:/stale/source\ntrain: images/train\nval: images/val\nnc: 1\nnames: {0: person}\n",
        encoding="utf-8",
    )

    dataset_root, data_file, images, labels, _ = MODULE.resolve_yolo_validation(root)

    assert dataset_root == root.resolve()
    assert data_file == (root / "data.yaml").resolve()
    assert images == (root / "images" / "val").resolve()
    assert labels == (root / "labels" / "val").resolve()


def test_resolve_yolo_validation_supports_split_first_layout(tmp_path):
    root = tmp_path / "dataset"
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "train: ./train/images\nval: ./valid/images\nnc: 2\nnames: [smoke, fire]\n",
        encoding="utf-8",
    )

    dataset_root, data_file, images, labels, _ = MODULE.resolve_yolo_validation(root)

    assert dataset_root == root.resolve()
    assert data_file == (root / "data.yaml").resolve()
    assert images == (root / "valid" / "images").resolve()
    assert labels == (root / "valid" / "labels").resolve()


def test_index_validation_excludes_images_with_invalid_boxes(tmp_path):
    image_root = tmp_path / "valid" / "images"
    label_root = tmp_path / "valid" / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    for name in ("valid", "invalid"):
        Image.new("RGB", (16, 12)).save(image_root / f"{name}.jpg")
    (label_root / "valid.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (label_root / "invalid.txt").write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="Excluded 1 validation image"):
        samples = MODULE.index_validation(
            image_root,
            label_root,
            {"nc": 2, "names": ["smoke", "fire"]},
        )

    assert [sample.relative_image.as_posix() for sample in samples] == ["valid.jpg"]


def test_cli_defaults_to_read_only_dry_run_and_dataset_calibration_output(tmp_path):
    args = MODULE.parse_args(["--data", str(tmp_path)])
    assert args.apply is False
    assert args.write_npz is False
    assert args.output is None
    assert MODULE.resolve_output_path(tmp_path, args.output) == (tmp_path / "calibration").resolve()


def test_output_path_allows_only_calibration_inside_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    calibration = (dataset / "calibration").resolve()

    assert MODULE.resolve_output_path(dataset, None) == calibration
    assert MODULE.resolve_output_path(dataset, dataset / "calibration") == calibration
    assert MODULE.resolve_output_path(dataset, tmp_path / "external") == (tmp_path / "external").resolve()

    with pytest.raises(MODULE.SelectionError, match="source dataset root"):
        MODULE.resolve_output_path(dataset, dataset)
    with pytest.raises(MODULE.SelectionError, match="Only .*calibration"):
        MODULE.resolve_output_path(dataset, dataset / "other")
