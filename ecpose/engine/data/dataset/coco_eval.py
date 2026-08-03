"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

from ...core import register
from ...misc import dist_utils
from ...misc.keypoint_sigmas import get_keypoint_sigmas

__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(
        self,
        coco_gt,
        iou_types,
        keypoint_sigmas=None,
        plate_metrics=False,
        plate_score_threshold=0.25,
        plate_iou_threshold=0.5,
        plate_miss_penalty=1.0,
    ):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types
        self.keypoint_sigmas = (
            keypoint_sigmas
            if keypoint_sigmas is not None
            else self._infer_keypoint_sigmas(coco_gt)
        )
        self.plate_metrics = plate_metrics
        self.plate_score_threshold = float(plate_score_threshold)
        self.plate_iou_threshold = float(plate_iou_threshold)
        # Normalised corner error charged to every ground-truth plate that no
        # prediction matched. 1.0 == the whole plate diagonal.
        self.plate_miss_penalty = float(plate_miss_penalty)
        self.plate_records = []
        self.plate_summary = {}

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = self._create_evaluator(coco_gt, iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = self._create_evaluator(self.coco_gt, iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        self.plate_records = []
        self.plate_summary = {}

    @staticmethod
    def _infer_keypoint_sigmas(coco_gt):
        counts = {
            len(category.get("keypoints", []))
            for category in coco_gt.dataset.get("categories", [])
            if category.get("keypoints")
        }
        if len(counts) != 1:
            return None
        try:
            return get_keypoint_sigmas(counts.pop())
        except ValueError:
            return None

    def _create_evaluator(self, coco_gt, iou_type):
        evaluator = COCOeval_faster(
            coco_gt, iouType=iou_type, print_function=print, separate_eval=True
        )
        if iou_type == "keypoints" and self.keypoint_sigmas is not None:
            evaluator.params.kpt_oks_sigmas = np.asarray(self.keypoint_sigmas, dtype=np.float32)
        return evaluator


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        if self.plate_metrics:
            self._update_plate_metrics(predictions)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

        if self.plate_metrics:
            gathered_records = dist_utils.all_gather(self.plate_records)
            records_by_image = {}
            for records in gathered_records:
                for record in records:
                    records_by_image[record["image_id"]] = record
            self.plate_records = [records_by_image[key] for key in sorted(records_by_image)]

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()
        if self.plate_metrics:
            self.plate_summary = self._summarize_plate_metrics()
            print(
                "Plate metrics: "
                f"LP-NME={self.plate_summary['plate_nme']:.6f}, "
                f"LP-NME(with misses)={self.plate_summary['plate_nme_penalized']:.6f}, "
                f"P95={self.plate_summary['plate_nme_p95']:.6f}, "
                f"precision={self.plate_summary['plate_precision']:.4f}, "
                f"recall={self.plate_summary['plate_recall']:.4f}, "
                f"F1={self.plate_summary['plate_f1']:.4f}"
            )

    @staticmethod
    def _keypoint_xy(keypoints):
        values = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
        return values[:, :2], values[:, 2] > 0

    @staticmethod
    def _keypoint_box(points):
        return np.asarray(
            [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
            dtype=np.float32,
        )

    @staticmethod
    def _box_iou(box1, box2):
        left = max(float(box1[0]), float(box2[0]))
        top = max(float(box1[1]), float(box2[1]))
        right = min(float(box1[2]), float(box2[2]))
        bottom = min(float(box1[3]), float(box2[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
        area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
        union = area1 + area2 - intersection
        return intersection / union if union > 0.0 else 0.0

    def _update_plate_metrics(self, predictions):
        for image_id, prediction in predictions.items():
            annotations = self.coco_gt.loadAnns(self.coco_gt.getAnnIds(imgIds=[image_id]))
            ground_truth = []
            for annotation in annotations:
                if annotation.get("iscrowd", 0):
                    continue
                points, visible = self._keypoint_xy(annotation["keypoints"])
                points = points[visible]
                if len(points) == 0:
                    continue
                ground_truth.append(
                    {
                        "label": int(annotation["category_id"]),
                        "points": points,
                        "box": self._keypoint_box(points),
                    }
                )

            scores = prediction["scores"].detach().cpu().numpy()
            labels = prediction["labels"].detach().cpu().numpy()
            keypoints = prediction["keypoints"].detach().cpu().numpy()
            selected = np.flatnonzero(scores >= self.plate_score_threshold)
            selected = selected[np.argsort(scores[selected])[::-1]]

            unmatched = set(range(len(ground_truth)))
            nmes = []
            max_corner_nmes = []
            for prediction_index in selected:
                pred_points, pred_visible = self._keypoint_xy(keypoints[prediction_index])
                pred_points = pred_points[pred_visible]
                if len(pred_points) == 0:
                    continue
                pred_box = self._keypoint_box(pred_points)
                candidates = [
                    gt_index
                    for gt_index in unmatched
                    if ground_truth[gt_index]["label"] == int(labels[prediction_index])
                ]
                if not candidates:
                    continue
                ious = [self._box_iou(pred_box, ground_truth[index]["box"]) for index in candidates]
                best_position = int(np.argmax(ious))
                if ious[best_position] < self.plate_iou_threshold:
                    continue

                gt_index = candidates[best_position]
                gt_points = ground_truth[gt_index]["points"]
                if pred_points.shape != gt_points.shape:
                    continue
                gt_box = ground_truth[gt_index]["box"]
                diagonal = float(np.hypot(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]))
                if diagonal <= 0.0:
                    continue
                corner_errors = np.linalg.norm(pred_points - gt_points, axis=1) / diagonal
                nmes.append(float(corner_errors.mean()))
                max_corner_nmes.append(float(corner_errors.max()))
                unmatched.remove(gt_index)

            self.plate_records.append(
                {
                    "image_id": int(image_id),
                    "ground_truth": len(ground_truth),
                    "predictions": len(selected),
                    "nmes": nmes,
                    "max_corner_nmes": max_corner_nmes,
                }
            )

    def _summarize_plate_metrics(self):
        ground_truth = sum(record["ground_truth"] for record in self.plate_records)
        predictions = sum(record["predictions"] for record in self.plate_records)
        nmes = np.asarray(
            [value for record in self.plate_records for value in record["nmes"]],
            dtype=np.float64,
        )
        max_corner_nmes = np.asarray(
            [value for record in self.plate_records for value in record["max_corner_nmes"]],
            dtype=np.float64,
        )
        matches = int(nmes.size)
        precision = matches / predictions if predictions else 0.0
        recall = matches / ground_truth if ground_truth else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

        def aggregate(values, operation):
            return float(operation(values)) if values.size else float("nan")

        # Mean corner error over every ground-truth plate, charging misses the
        # full penalty. Unlike `plate_nme` this cannot be gamed by predicting
        # only the few easy plates, so it is safe to select checkpoints on.
        if ground_truth:
            penalized_nme = float(
                (nmes.sum() + self.plate_miss_penalty * (ground_truth - matches)) / ground_truth
            )
        else:
            penalized_nme = float("nan")

        return {
            "plate_nme_penalized": penalized_nme,
            "plate_nme": aggregate(nmes, np.mean),
            "plate_nme_median": aggregate(nmes, np.median),
            "plate_nme_p95": aggregate(nmes, lambda values: np.percentile(values, 95)),
            "plate_max_corner_nme": aggregate(max_corner_nmes, np.mean),
            "plate_max_corner_nme_p95": aggregate(
                max_corner_nmes, lambda values: np.percentile(values, 95)
            ),
            "plate_precision": float(precision),
            "plate_recall": float(recall),
            "plate_f1": float(f1),
            "plate_matches": float(matches),
            "plate_ground_truth": float(ground_truth),
            "plate_predictions": float(predictions),
        }

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
                
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
