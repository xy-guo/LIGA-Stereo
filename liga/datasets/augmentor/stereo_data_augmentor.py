# data augmentor for stereo data_dict.

from functools import partial
import numpy as np

from liga.utils import common_utils, box_utils


class StereoDataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []

        if augmentor_configs is not None:
            aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) else augmentor_configs.AUG_CONFIG_LIST

            for cur_cfg in aug_config_list:
                if not isinstance(augmentor_configs, list):
                    if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                        continue
                if cur_cfg.NAME in ["gt_sampling"]:
                    cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
                else:
                    cur_augmentor = partial(getattr(self, cur_cfg.NAME), config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)

    def pre_2d_transformation(self, data_dict):
        assert 'did_3d_transformation' not in data_dict
        assert 'gt_boxes_no3daug' not in data_dict
        assert 'points_no3daug' not in data_dict

    def pre_world_transformation(self, data_dict):
        data_dict['did_3d_transformation'] = True
        if 'gt_boxes_no3daug' not in data_dict:
            data_dict['gt_boxes_no3daug'] = data_dict['gt_boxes'].copy()
        if 'points_no3daug' not in data_dict:
            data_dict['points_no3daug'] = data_dict['points'].copy()

    def random_crop(self, data_dict, config=None):
        self.pre_2d_transformation(data_dict)

        crop_rel_x = np.random.uniform(low=config.MIN_REL_X, high=config.MAX_REL_X) / 2 + 0.5
        crop_rel_y = np.random.uniform(low=config.MIN_REL_Y, high=config.MAX_REL_Y) / 2 + 0.5
        old_h, old_w = data_dict['left_img'].shape[:2]
        crop_h, crop_w = min(config.MAX_CROP_H, old_h), min(config.MAX_CROP_W, old_w)
        assert crop_h <= old_h and crop_w <= old_w and 0 <= crop_rel_x <= 1 and 0 <= crop_rel_y <= 1

        x1 = int((old_w - crop_w) * crop_rel_x)
        y1 = int((old_h - crop_h) * crop_rel_y)

        data_dict['left_img'] = data_dict['left_img'][y1: y1 + crop_h, x1:x1 + crop_w]
        data_dict['right_img'] = data_dict['right_img'][y1: y1 + crop_h, x1:x1 + crop_w]
        data_dict['calib'].offset(x1, y1)
        if 'image_shape' in data_dict:
            data_dict['image_shape'] = data_dict['left_img'].shape[:2]
        if 'gt_boxes_2d_ignored' in data_dict:
            data_dict['gt_boxes_2d_ignored'] = data_dict['gt_boxes_2d_ignored'].copy()
            data_dict['gt_boxes_2d_ignored'][:, [0, 2]] -= x1
            data_dict['gt_boxes_2d_ignored'][:, [1, 3]] -= y1
        return data_dict

    def filter_truncated(self, data_dict, config=None):
        assert 'gt_boxes' in data_dict, 'should not call filter_truncated in test mode'
        self.pre_2d_transformation(data_dict)

        # reproject bboxes into image space and do filtering by truncated ratio
        area_ratio_threshold = config.AREA_RATIO_THRESH
        area_2d_ratio_threshold = config.AREA_2D_RATIO_THRESH
        gt_truncated_threshold = config.GT_TRUNCATED_THRESH

        valid_mask = data_dict['gt_boxes_mask'][data_dict['gt_boxes_mask']]
        if area_ratio_threshold is not None:
            assert area_ratio_threshold >= 0.9, 'AREA_RATIO_THRESH should be >= 0.9'
            image_shape = data_dict['left_img'].shape[:2]
            calib = data_dict['calib']
            gt_boxes_cam = box_utils.boxes3d_lidar_to_kitti_camera(data_dict['gt_boxes'][data_dict['gt_boxes_mask']], None, pseudo_lidar=True)

            boxes2d_image, _ = box_utils.boxes3d_kitti_camera_to_imageboxes(gt_boxes_cam, calib, image_shape, return_neg_z_mask=True, fix_neg_z_bug=True)
            truncated_ratio = 1 - box_utils.boxes3d_kitti_camera_inside_image_mask(gt_boxes_cam, calib, image_shape, reduce=False).mean(-1)
            valid_mask &= truncated_ratio < area_ratio_threshold

        if area_2d_ratio_threshold is not None:
            assert area_2d_ratio_threshold >= 0.9, 'AREA_2D_RATIO_THRESH should be >= 0.9'
            image_shape = data_dict['left_img'].shape[:2]
            boxes2d_image, no_neg_z_valids = box_utils.boxes3d_kitti_camera_to_imageboxes(
                box_utils.boxes3d_lidar_to_kitti_camera(data_dict['gt_boxes'][data_dict['gt_boxes_mask']], data_dict['calib'], pseudo_lidar=True),
                data_dict['calib'],
                return_neg_z_mask=True,
                fix_neg_z_bug=True
            )
            boxes2d_inside = np.zeros_like(boxes2d_image)
            boxes2d_inside[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_inside[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
            boxes2d_inside[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
            boxes2d_inside[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)
            clip_box_area = (boxes2d_inside[:, 2] - boxes2d_inside[:, 0]) * (boxes2d_inside[:, 3] - boxes2d_inside[:, 1])
            full_box_area = (boxes2d_image[:, 2] - boxes2d_image[:, 0]) * (boxes2d_image[:, 3] - boxes2d_image[:, 1])
            clip_ratio = 1 - clip_box_area / full_box_area
            valid_mask &= clip_ratio < area_2d_ratio_threshold

        if gt_truncated_threshold is not None:
            gt_truncated = data_dict['gt_truncated'][data_dict['gt_boxes_mask']]
            valid_mask &= gt_truncated < gt_truncated_threshold

        cared_mask = data_dict['gt_boxes_mask'].copy()
        if not all(valid_mask):
            invalid_mask = ~valid_mask
            print(config)
            print('filter truncated ratio:', truncated_ratio[invalid_mask] if area_ratio_threshold is not None else 'null',
                  '3d boxes', data_dict['gt_boxes'][cared_mask][invalid_mask],
                  'flipped', data_dict['calib'].flipped,
                  'image idx', data_dict['image_idx'],
                  'frame_id', data_dict['frame_id'],
                  '\n')

        data_dict['gt_boxes_mask'][cared_mask] = valid_mask
        return data_dict

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_rotation(self, data_dict, config=None):
        self.pre_world_transformation(data_dict)

        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, T = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range,
            return_trans_mat=True
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points  # points_for2d is fixed since images do not support rotation
        # note that random T is the inverse transformation matrix
        data_dict['random_T'] = np.matmul(data_dict.get('random_T', np.eye(4)), T)

        return data_dict

    def random_world_scaling(self, data_dict, config=None):
        self.pre_world_transformation(data_dict)

        gt_boxes, points, T = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'],
            return_trans_mat=True
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        # note that random T is the inverse transformation matrix
        data_dict['random_T'] = np.matmul(data_dict.get('random_T', np.eye(4)), T)

        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        self.pre_world_transformation(data_dict)

        gt_boxes, points, T = augmentor_utils.global_translation(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_TRANSLATION_RANGE'],
            return_trans_mat=True
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        # note that random T is the inverse transformation matrix
        data_dict['random_T'] = np.matmul(data_dict.get('random_T', np.eye(4)), T)

        return data_dict

    def forward(self, data_dict):
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        if 'gt_boxes' in data_dict:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            for key in ['gt_names', 'gt_boxes', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index']:
                data_dict[key] = data_dict[key][gt_boxes_mask]
            for key in ['gt_boxes_no3daug']:
                if key in data_dict:
                    data_dict[key] = data_dict[key][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')

        return data_dict
