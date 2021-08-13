# Stereo KITTI Pytorch Dataset (for training Our LIGA model)

import copy
import pickle
import numpy as np
import torch
from skimage import io

from liga.utils import box_utils, calibration_kitti, common_utils, object3d_kitti, depth_map_utils
from liga.datasets.stereo_dataset_template import StereoDatasetTemplate


class StereoKittiDataset(StereoDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.flip = self.dataset_cfg.FLIP
        self.force_flip = getattr(self.dataset_cfg, 'FORCE_FLIP', False)
        self.boxes_gt_in_cam2_view = getattr(self.dataset_cfg, 'BOXES_GT_IN_CAM2_VIEW', False)
        self.use_van = self.dataset_cfg.USE_VAN and training
        self.use_person_sitting = self.dataset_cfg.USE_PERSON_SITTING and training
        self.cat_reflect = self.dataset_cfg.CAT_REFLECT_DIM

        logger.info('boxes_gt_in_cam2_view %s' % self.boxes_gt_in_cam2_view)

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(
            split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)
        assert len(self.sample_id_list) == len(self.kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' %
                             (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / \
            ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(
            split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists(), f"{lidar_file} not found"
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_image(self, idx, image_id=2):
        img_file = self.root_split_path / \
            ('image_%s' % image_id) / ('%s.png' % idx)
        assert img_file.exists()
        return io.imread(img_file).copy()

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(
            pts_img[:, 0] > 0, pts_img[:, 0] < img_shape[1] - 1)
        val_flag_2 = np.logical_and(
            pts_img[:, 1] > 0, pts_img[:, 1] < img_shape[0] - 1)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None, mode_2d=False):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib_aug = batch_dict['calib'][batch_index]
            calib_ori = batch_dict['calib_ori'][batch_index] if 'calib_ori' in batch_dict else calib_aug
            image_shape = batch_dict['image_shape'][batch_index]
            # NOTE: in stereo mode, the 3d boxes are predicted in pseudo lidar coordinates
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(
                pred_boxes, None, pseudo_lidar=True, pseduo_cam2_view=self.boxes_gt_in_cam2_view)
            # only for debug, calib.flipped should be False when testing
            if calib_aug.flipped:
                pred_boxes_camera = box_utils.boxes3d_fliplr(pred_boxes_camera, cam_view=True)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib_ori, image_shape=image_shape,
                fix_neg_z_bug=True
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            # pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        def generate_single_2d_sample_dict(batch_index, box_dict):
            def to_numpy(x):
                if isinstance(x, np.ndarray):
                    return x
                elif isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                else:
                    raise ValueError('wrong type of input')
            pred_scores_2d = to_numpy(box_dict['pred_scores_2d'])
            pred_boxes_2d = to_numpy(box_dict['pred_boxes_2d'])
            pred_labels_2d = to_numpy(box_dict['pred_labels_2d'])
            pred_dict = get_template_prediction(pred_scores_2d.shape[0])
            calib = batch_dict['calib'][batch_index]
            # calib_ori = batch_dict['calib_ori'][batch_index] if 'calib_ori' in batch_dict else calib
            if pred_scores_2d.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels_2d - 1]
            pred_dict['bbox'] = pred_boxes_2d[:, :4]

            pred_dict['bbox'][:, [0, 2]] += calib.offsets[0]
            pred_dict['bbox'][:, [1, 3]] += calib.offsets[1]

            pred_dict['score'] = pred_scores_2d

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            if not mode_2d:
                single_pred_dict = generate_single_sample_dict(index, box_dict)
            else:
                single_pred_dict = generate_single_2d_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.8f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, eval_metric='3d', **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]

        if eval_metric == '2d':
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result_2d(
                eval_gt_annos, eval_det_annos, class_names)
        else:
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        assert self.dataset_cfg.FOV_POINTS_ONLY
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        if not self.force_flip:
            if self.flip and self.mode == 'train':
                flip_this_image = np.random.randint(2) > 0.5
            else:
                flip_this_image = False
        else:
            flip_this_image = True

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        raw_points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)
        calib_ori = copy.deepcopy(calib)

        pts_rect = calib.lidar_to_rect(raw_points[:, 0:3])
        reflect = raw_points[:, 3:4]
        if flip_this_image:
            calib.fliplr(info['image']['image_shape'][1])
            pts_rect[:, 0] *= -1

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            pts_rect = pts_rect[fov_flag]
            reflect = reflect[fov_flag]

        # load images
        left_img = self.get_image(info['image']['image_idx'], 2)
        right_img = self.get_image(info['image']['image_idx'], 3)

        if flip_this_image:
            right_img, left_img = left_img[:, ::-1], right_img[:, ::-1]

        # convert camera-view points into pseudo lidar points
        # see code in calibration_kitti.py
        # right: [x] --> [-y]
        # up: [-y] --> [z]
        # front: [z] --> [x]
        if self.cat_reflect:
            input_points = np.concatenate([calib.rect_to_lidar_pseudo(pts_rect), reflect], 1)
        else:
            input_points = calib.rect_to_lidar_pseudo(pts_rect)
        input_dict = {
            'points': input_points,
            'frame_id': sample_idx,
            'calib': calib,
            'calib_ori': calib_ori,
            'left_img': left_img,
            'right_img': right_img,
            'image_shape': left_img.shape
        }

        if 'annos' in info:
            annos = info['annos']
            if self.use_van:
                # Car 14357, Van 1297
                annos['name'][annos['name'] == 'Van'] = 'Car'
            if self.use_person_sitting:
                # Ped 2207, Person_sitting 56
                annos['name'][annos['name'] == 'Person_sitting'] = 'Pedestrian'
            full_annos = annos
            ignored_annos = common_utils.collect_ignored_with_name(full_annos, name=['DontCare'])  # only bbox is useful
            annos = common_utils.drop_info_with_name(full_annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_2d_ignored = ignored_annos['bbox']
            gt_truncated = annos['truncated']
            gt_occluded = annos['occluded']
            gt_difficulty = annos['difficulty']
            gt_index = annos['index']
            image_shape = left_img.shape

            if flip_this_image:
                gt_boxes_camera = box_utils.boxes3d_fliplr(gt_boxes_camera, cam_view=True)
                gt_boxes_2d_ignored = box_utils.boxes2d_fliplr(gt_boxes_2d_ignored, image_shape)

            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(
                gt_boxes_camera, calib, pseudo_lidar=True, pseudo_cam2_view=self.boxes_gt_in_cam2_view)
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_boxes_2d_ignored': gt_boxes_2d_ignored,
                'gt_truncated': gt_truncated,
                'gt_occluded': gt_occluded,
                'gt_difficulty': gt_difficulty,
                'gt_index': gt_index,
                'image_idx': index
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict
