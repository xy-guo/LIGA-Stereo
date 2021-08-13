import numpy as np


def points_to_depth_map(pts_rect, img_shape, calib):
    depth_gt_img = np.zeros(img_shape, dtype=np.float32)
    pts_img, pts_depth = calib.rect_to_img(pts_rect[:, :3])
    iy, ix = np.round(pts_img[:, 1]).astype(np.int64), np.round(pts_img[:, 0]).astype(np.int64)
    mask = (iy >= 0) & (ix >= 0) & (iy < depth_gt_img.shape[0]) & (ix < depth_gt_img.shape[1])
    iy, ix = iy[mask], ix[mask]
    depth_gt_img[iy, ix] = pts_depth[mask]
    return depth_gt_img
