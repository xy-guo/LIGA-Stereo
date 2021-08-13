# Modified from OpenPCDet. https://github.com/open-mmlab/OpenPCDet
# Augmentation utility functions.

import numpy as np

from liga.utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, return_trans_mat=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    if not return_trans_mat:
        return gt_boxes, points
    else:
        T = np.eye(4, dtype=np.float32)
        T[0, 0] = T[1, 1] = np.cos(noise_rotation)
        T[0, 1] = np.sin(noise_rotation)
        T[1, 0] = -np.sin(noise_rotation)
        return gt_boxes, points, T


def global_scaling(gt_boxes, points, scale_range, return_trans_mat=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if not return_trans_mat:
        return gt_boxes, points
    else:
        T = np.eye(4, dtype=np.float32)
        T[0, 0] = 1 / noise_scale
        T[1, 1] = 1 / noise_scale
        T[2, 2] = 1 / noise_scale
        return gt_boxes, points, T


def global_translation(gt_boxes, points, translation_range, return_trans_mat=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
    """
    # in lidar coordinate
    tx = np.random.uniform(low=translation_range[0], high=translation_range[3])
    ty = np.random.uniform(low=translation_range[1], high=translation_range[4])
    tz = np.random.uniform(low=translation_range[2], high=translation_range[5])
    gt_boxes[:, 0] += tx
    gt_boxes[:, 1] += ty
    gt_boxes[:, 2] += tz
    points[:, 0] += tx
    points[:, 1] += ty
    points[:, 2] += tz

    if not return_trans_mat:
        return gt_boxes, points
    else:
        T = np.eye(4, dtype=np.float32)
        T[0, 3] -= tx
        T[1, 3] -= ty
        T[2, 3] -= tz
        return gt_boxes, points, T
