"""Shared OKS sigma definitions for supported keypoint layouts."""

import numpy as np


def get_keypoint_sigmas(num_keypoints: int) -> np.ndarray:
    if num_keypoints == 17:
        values = [
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
            1.07, 1.07, .87, .87, .89, .89,
        ]
        return np.asarray(values, dtype=np.float32) / 10.0
    if num_keypoints == 14:
        values = [.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]
        return np.asarray(values, dtype=np.float32) / 10.0
    if num_keypoints == 4:
        # Uniform tolerance for LT, RT, RB and LB plate corners.
        return np.full(4, 0.05, dtype=np.float32)
    if num_keypoints == 3:
        return np.asarray([1.07, 1.07, .67], dtype=np.float32) / 10.0
    raise ValueError(f"Unsupported keypoints number {num_keypoints}")
