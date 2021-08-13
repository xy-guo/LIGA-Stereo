from collections import defaultdict
from pathlib import Path
import numpy as np
import torch.utils.data as torch_data

from liga.utils import common_utils, box_utils, depth_map_utils
from liga.ops.roiaware_pool3d import roiaware_pool3d_utils
from .augmentor.stereo_data_augmentor import StereoDataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
from liga.utils.calibration_kitti import Calibration


class StereoDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(
            self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(
            self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.voxel_size = self.dataset_cfg.VOXEL_SIZE
        grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

        if self.dataset_cfg.get("STEREO_VOXEL_SIZE", None):
            self.stereo_voxel_size = self.dataset_cfg.STEREO_VOXEL_SIZE
            stereo_grid_size = (
                self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.stereo_voxel_size)
            self.stereo_grid_size = np.round(stereo_grid_size).astype(np.int64)

        if self.training:
            self.data_augmentor = StereoDataAugmentor(
                self.root_path, self.dataset_cfg.TRAIN_DATA_AUGMENTOR, self.class_names, logger=self.logger
            )
        else:
            if getattr(self.dataset_cfg, 'TEST_DATA_AUGMENTOR', None) is not None:
                self.data_augmentor = StereoDataAugmentor(
                    self.root_path, self.dataset_cfg.TEST_DATA_AUGMENTOR, self.class_names, logger=self.logger
                )
                # logger.warn('using data augmentor in test mode')
            else:
                self.data_augmentor = None

        if self.dataset_cfg.get('POINT_FEATURE_ENCODING'):
            self.point_feature_encoder = PointFeatureEncoder(
                self.dataset_cfg.POINT_FEATURE_ENCODING,
                point_cloud_range=self.point_cloud_range
            )
        else:
            self.point_feature_encoder = None
        if self.dataset_cfg.get('DATA_PROCESSOR'):
            self.data_processor = DataProcessor(
                self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
            )
        else:
            self.data_processor = None

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array(
                [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            # TODO: in case using data augmentor, please pay attention to the coordinate
            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

            if self.training and len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        elif (not self.training) and self.data_augmentor:
            # only do some basic image scaling and cropping
            data_dict = self.data_augmentor.forward(data_dict)

        if data_dict.get('gt_boxes', None) is not None:
            if 'gt_boxes_no3daug' not in data_dict:
                data_dict['gt_boxes_no3daug'] = data_dict['gt_boxes'].copy()

            selected = common_utils.keep_arrays_by_name(
                data_dict['gt_names'], self.class_names)
            if len(selected) != len(data_dict['gt_names']):
                for key in ['gt_names', 'gt_boxes', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index', 'gt_boxes_no3daug']:
                    data_dict[key] = data_dict[key][selected]
            gt_classes = np.array([self.class_names.index(
                n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            data_dict['gt_boxes'] = np.concatenate(
                (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes_no3daug'] = np.concatenate(
                (data_dict['gt_boxes_no3daug'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

        # convert to 2d gt boxes
        image_shape = data_dict['left_img'].shape[:2]
        if 'gt_boxes' in data_dict:
            gt_boxes_no3daug = data_dict['gt_boxes_no3daug']
            gt_boxes_no3daug_cam = box_utils.boxes3d_lidar_to_kitti_camera(gt_boxes_no3daug, None, pseudo_lidar=True)
            data_dict['gt_boxes_2d'] = box_utils.boxes3d_kitti_camera_to_imageboxes(
                gt_boxes_no3daug_cam, data_dict['calib'], image_shape, fix_neg_z_bug=True)
            data_dict['gt_centers_2d'] = box_utils.boxes3d_kitti_camera_to_imagecenters(
                gt_boxes_no3daug_cam, data_dict['calib'], image_shape)

        if self.point_feature_encoder:
            data_dict = self.point_feature_encoder.forward(data_dict)
        if self.data_processor:
            data_dict = self.data_processor.forward(data_dict=data_dict)

        # generate depth gt image
        rect_points = Calibration.lidar_pseudo_to_rect(data_dict.get('points_no3daug', data_dict['points'])[:, :3])
        data_dict['depth_gt_img'] = depth_map_utils.points_to_depth_map(rect_points, image_shape, data_dict['calib'])
        if 'gt_boxes' in data_dict:
            data_dict['depth_fgmask_img'] = roiaware_pool3d_utils.depth_map_in_boxes_cpu(
                data_dict['depth_gt_img'], data_dict['gt_boxes'][:, :7], data_dict['calib'], expand_distance=0., expand_ratio=1.0)

        data_dict.pop('points_no3daug', None)
        data_dict.pop('did_3d_transformation', None)
        data_dict.pop('road_plane', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                ret[key] = np.concatenate(val, axis=0)
            elif key in ['points', 'voxel_coords']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['left_img', 'right_img', 'depth_gt_img', 'depth_fgmask_img']:
                if key in ['depth_gt_img', 'depth_fgmask_img']:
                    val = [np.expand_dims(x, -1) for x in val]
                max_h = np.max([x.shape[0] for x in val])
                max_w = np.max([x.shape[1] for x in val])
                pad_h = (max_h - 1) // 32 * 32 + 32 - max_h
                pad_w = (max_w - 1) // 32 * 32 + 32 - max_w
                assert pad_h < 32 and pad_w < 32
                padded_imgs = []
                for i, img in enumerate(val):
                    if key in ['left_img', 'right_img']:
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img = (img.astype(np.float32) / 255 - mean) / std
                    img = np.pad(img, ((0, pad_h), (0, pad_w),
                                       (0, 0)), mode='constant')
                    padded_imgs.append(img)
                ret[key] = np.stack(
                    padded_imgs, axis=0).transpose(0, 3, 1, 2)
            elif key in ['gt_boxes', 'gt_boxes_no3daug', 'gt_boxes_2d', 'gt_centers_2d', 'gt_boxes_2d_ignored', 'gt_boxes_camera']:
                max_gt = max([len(x) for x in val])
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                ret[key] = batch_gt_boxes3d
            elif key in ['image_idx']:  # gt_boxes_mask
                ret[key] = val
            elif key in ['gt_names', 'gt_truncated', 'gt_occluded', 'gt_difficulty', 'gt_index']:
                ret[key] = [np.array(x) for x in val]
            elif key in ['calib', 'calib_ori', 'use_lead_xyz']:
                ret[key] = val
            elif key in ['frame_id', 'image_shape', 'random_T']:
                ret[key] = np.stack(val, axis=0)
            else:
                print(key)
                raise NotImplementedError

        ret['batch_size'] = batch_size
        return ret
