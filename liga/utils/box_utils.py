from liga.utils.calibration_kitti import Calibration
import numpy as np
import scipy
import torch
from scipy.spatial import Delaunay

from liga.ops.roiaware_pool3d import roiaware_pool3d_utils
from liga.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
from . import common_utils


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib, pseudo_lidar=False, pseudo_cam2_view=False):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    xyz_camera = boxes3d_camera[:, 0:3]
    l, h, w, r = boxes3d_camera[:, 3:4], boxes3d_camera[:, 4:5], boxes3d_camera[:, 5:6], boxes3d_camera[:, 6:7]
    if not pseudo_lidar:
        assert calib is not None, "calib can only be none when pseudo_lidar is True"
        xyz_lidar = calib.rect_to_lidar(xyz_camera)
    else:
        if pseudo_cam2_view:
            xyz_camera = xyz_camera + calib.txyz
        xyz_lidar = Calibration.rect_to_lidar_pseudo(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_fliplr(boxes3d, cam_view=True):
    if cam_view:
        alpha = boxes3d[:, 6:7]
        alpha = np.pi-alpha  # ((alpha > 0).astype(np.float) + (alpha <= 0).astype(np.float) * -1) * np.pi - alpha
        return np.concatenate([-boxes3d[:, 0:1], boxes3d[:, 1:6], alpha], axis=1)
    else:
        raise NotImplementedError


def boxes2d_fliplr(boxes2d, image_shape):
    x1, y1, x2, y2 = boxes2d[:, 0], boxes2d[:, 1], boxes2d[:, 2], boxes2d[:, 3]
    img_w = image_shape[1]
    return np.stack([img_w - 1 - x1, y1, img_w - 1 - x2, y2], axis=1)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib=None, pseudo_lidar=False, pseduo_cam2_view=False):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    # TODO: will modify original boxes3d_lidar
    xyz_lidar = boxes3d_lidar[:, 0:3].copy()
    l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    if not pseudo_lidar:
        assert calib is not None, "calib can only be None in pseudo_lidar mode"
        xyz_cam = calib.lidar_to_rect(xyz_lidar)
    else:
        # transform xyz from pseudo-lidar to camera view
        xyz_cam = Calibration.lidar_pseudo_to_rect(xyz_lidar)
        if pseduo_cam2_view:
            xyz_cam = xyz_cam - calib.txyz
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_to_grid3d_kitti_camera(boxes3d, size=28, bottom_center=True, surface=False):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners, y_corners, z_corners = np.meshgrid(np.linspace(-0.5, 0.5, size), np.linspace(-0.5, 0.5, size), np.linspace(-0.5, 0.5, size))
    if surface:
        surface_mask = (np.abs(x_corners) == 0.5) | (np.abs(y_corners) == 0.5) | (np.abs(z_corners) == 0.5)
        x_corners = x_corners[surface_mask]
        y_corners = y_corners[surface_mask]
        z_corners = z_corners[surface_mask]
    x_corners = x_corners.reshape([1, -1]) * l.reshape([-1, 1])
    y_corners = y_corners.reshape([1, -1]) * h.reshape([-1, 1])
    z_corners = z_corners.reshape([1, -1]) * w.reshape([-1, 1])
    if bottom_center:
        y_corners -= h.reshape([-1, 1]) / 2

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.stack([x_corners, y_corners, z_corners], axis=-1)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners
    y = y_loc.reshape(-1, 1) + y_corners
    z = z_loc.reshape(-1, 1) + z_corners

    corners = np.stack([x, y, z], axis=-1)

    return corners.astype(np.float32)


def torch_boxes3d_to_corners3d_kitti_lidar(boxes3d):
    """
    :param boxes3d: (N, ..., 7) [x, y, z, l, w, h, rz] in lidar coords
    :param bottom_center: whether z is on the bottom center of object
    :return: corners3d: (N, ..., 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    l, w, h = boxes3d[..., 3], boxes3d[..., 4], boxes3d[..., 5]  # [...]
    x_corners = torch.stack([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dim=-1)  # [..., 8]
    y_corners = torch.stack([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=-1)  # [..., 8]
    z_corners = torch.stack([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dim=-1)  # [..., 8]

    ry = boxes3d[..., 6]
    zeros, ones = torch.zeros_like(ry), torch.ones_like(ry)
    cosy, siny = torch.cos(ry), torch.sin(ry)
    R_list = torch.stack([cosy, siny, zeros,
                          -siny, cosy, zeros,
                          zeros, zeros, ones], dim=-1)
    R_list = R_list.view(*R_list.shape[:-1], 3, 3)  # (..., 3, 3)

    temp_corners = torch.stack([x_corners, y_corners, z_corners], dim=-1)  # (..., 8, 3)
    rotated_corners = torch.matmul(temp_corners, R_list)  # (..., 8, 3)

    loc = boxes3d[..., :3].unsqueeze(-2)
    rotated_corners = rotated_corners + loc

    return rotated_corners


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None, return_neg_z_mask=False, fix_neg_z_bug=False):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    if not fix_neg_z_bug:
        corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
        pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
        corners_in_image = pts_img.reshape(-1, 8, 2)

        min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
        max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
        boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        if not return_neg_z_mask:
            return boxes2d_image
        else:
            return boxes2d_image, np.all(corners3d[:, :, 2] >= 0.01, axis=1)
    else:
        num_boxes = boxes3d.shape[0]
        corners3d = boxes3d_to_grid3d_kitti_camera(boxes3d, size=7, surface=False)
        if num_boxes != 0:
            num_points = corners3d.shape[1]
            pts_img, pts_depth = calib.rect_to_img(corners3d.reshape(-1, 3))
            corners_in_image = pts_img.reshape(num_boxes, num_points, 2)
            depth_in_image = pts_depth.reshape(num_boxes, num_points)

            min_uv = np.array([np.min(x[d > 0], axis=0) for x, d in zip(corners_in_image, depth_in_image)])  # (N, 2)
            max_uv = np.array([np.max(x[d > 0], axis=0) for x, d in zip(corners_in_image, depth_in_image)])  # (N, 2)
            boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
        else:
            boxes2d_image = np.zeros([0, 4], dtype=np.float32)

        if image_shape is not None:
            boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

        if not return_neg_z_mask:
            return boxes2d_image
        else:
            return boxes2d_image, np.all(corners3d[:, :, 2] >= 0.01, axis=1)


def boxes3d_kitti_camera_inside_image_mask(boxes3d, calib, image_shape, reduce=True):
    corners3d = boxes3d_to_grid3d_kitti_camera(boxes3d, size=28, surface=True)
    num_points = corners3d.shape[1]
    pts_img, pts_depth = calib.rect_to_img(corners3d.reshape(-1, 3))
    pts_img = pts_img.reshape(-1, num_points, 2)
    pts_u, pts_v = pts_img[..., 0], pts_img[..., 1]
    pts_depth = pts_depth.reshape(-1, num_points)

    valid_depth = pts_depth > 0
    valid_in_image = (pts_u > 0) & (pts_v > 0) & (pts_u < image_shape[1]) & (pts_v < image_shape[0])

    valid_mask = valid_depth & valid_in_image

    if reduce:
        return np.any(valid_mask, 1)
    else:
        return valid_mask


def boxes3d_kitti_camera_to_imagecenters(boxes3d, calib, image_shape=None):
    centers3d = boxes3d_to_corners3d_kitti_camera(boxes3d).mean(1)

    pts_img, _ = calib.rect_to_img(centers3d)

    return pts_img


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def boxes3d_direction_aligned_bev_iou(boxes_a, boxes_b, angle_threshold=np.pi / 4, angle_cycle=np.pi):
    """
    This function is similar to boxes3d_nearest_bev_iou.
    The directions of anchor boxes (boxes_a) are aligned using its nearest gt box,
    When the angle difference is larger than angle_threshold,

    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ious
    """
    # find the bev centers for boxes a and b
    center_bev_a = boxes_a[:, 0:2]
    center_bev_b = boxes_b[:, 0:2]

    # compute pairwise distance to compute the nearest gt indexes
    center_dist = torch.norm(center_bev_a[:, None, :] - center_bev_b[None, :, :], dim=-1)
    nearest_gt_ids = center_dist.argmin(1)

    # compute pairwise angle difference and limit by period pi
    angle_dist = boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :]
    angle_dist = common_utils.limit_period(angle_dist, offset=0.5, period=angle_cycle)
    assert torch.all(angle_dist > -angle_cycle / 2 - 1e-4)
    assert torch.all(angle_dist < angle_cycle / 2 + 1e-4)
    angle_dist = angle_dist.abs()

    # use the nearest gt indexes to align the anchor boxes,
    # then compute the iou
    aligned_boxes_a = torch.cat([boxes_a[:, :3], boxes_b[:, 3:7][nearest_gt_ids]], dim=1)
    # aligned_boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(aligned_boxes_a)
    # boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)
    iou = boxes_iou_bev(aligned_boxes_a, boxes_b)

    # force >angle thresh to be zero
    iou[angle_dist > angle_threshold] = 0.

    return iou
