"""A variant of anchor_head_single.

The differences are as follows:
* two more options: num_convs, GN
* apply two split convs for regression outputs and classification outputs
* when num_convs == 0, this module should be almost the same as anchor_head_single
* in conv_box/cls, the kernel size is modified to 3 instead of 1
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation, gn=False, groups=32):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(groups, out_planes))


class DetHead(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        self.num_convs = model_cfg.NUM_CONVS
        self.GN = model_cfg.GN
        self.xyz_for_angles = model_cfg.xyz_for_angles
        self.hwl_for_angles = model_cfg.hwl_for_angles

        if self.num_convs > 0:
            self.rpn3d_cls_convs = []
            self.rpn3d_bbox_convs = []
            for _ in range(self.num_convs):
                self.rpn3d_cls_convs.append(
                    nn.Sequential(
                        convbn(input_channels, input_channels, 3, 1, 1, 1, gn=self.GN),
                        nn.ReLU(inplace=True))
                )
                self.rpn3d_bbox_convs.append(
                    nn.Sequential(
                        convbn(input_channels, input_channels, 3, 1, 1, 1, gn=self.GN),
                        nn.ReLU(inplace=True))
                )
            assert len(self.rpn3d_cls_convs) == self.num_convs
            assert len(self.rpn3d_bbox_convs) == self.num_convs
            self.rpn3d_cls_convs = nn.Sequential(*self.rpn3d_cls_convs)
            self.rpn3d_bbox_convs = nn.Sequential(*self.rpn3d_bbox_convs)

        cls_feature_channels = input_channels
        cls_groups = 1

        self.conv_cls = nn.Conv2d(
            cls_feature_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=3, padding=1, stride=1, groups=cls_groups
        )
        if self.xyz_for_angles and self.hwl_for_angles:
            box_dim = self.num_anchors_per_location * self.box_coder.code_size
        elif not self.xyz_for_angles and not self.hwl_for_angles:
            box_dim = self.num_class * 6 + self.num_anchors_per_location * (self.box_coder.code_size - 6)
        else:
            box_dim = self.num_class * 3 + self.num_anchors_per_location * (self.box_coder.code_size - 3)
        self.conv_box = nn.Conv2d(
            input_channels, box_dim,
            kernel_size=3, padding=1, stride=1
        )

        self.num_angles = self.num_anchors_per_location // self.num_class

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                cls_feature_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
                groups=cls_groups
            )
        else:
            self.conv_dir_cls = None

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.normal_(self.conv_cls.weight, std=0.1)
        nn.init.normal_(self.conv_box.weight, std=0.02)
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # NOTE: clear forward ret dict to avoid potential bugs
        self.forward_ret_dict.clear()

        spatial_features_2d = data_dict['spatial_features_2d']

        if self.do_feature_imitation and self.training:
            if 'gt_boxes' in data_dict:
                self.forward_ret_dict['gt_boxes'] = data_dict['gt_boxes']
            self.forward_ret_dict['imitation_features_pairs'] = []
            imitation_conv_layers = [self.conv_imitation] if len(self.imitation_configs) == 1 else self.conv_imitation
            for cfg, imitation_conv in zip(self.imitation_configs, imitation_conv_layers):
                lidar_feature_name = cfg.lidar_feature_layer
                stereo_feature_name = cfg.stereo_feature_layer
                self.forward_ret_dict['imitation_features_pairs'].append(
                    dict(
                        config=cfg,
                        stereo_feature_name=stereo_feature_name,
                        lidar_feature_name=lidar_feature_name,
                        gt=data_dict['lidar_outputs'][lidar_feature_name],
                        pred=imitation_conv(data_dict[stereo_feature_name])
                    )
                )

            # for k in data_dict:
            #     if k in ["lidar_batch_cls_preds", "lidar_batch_box_preds"]:
            #         self.forward_ret_dict[k] = data_dict[k]

        cls_features = spatial_features_2d
        reg_features = spatial_features_2d
        if self.num_convs > 0:
            cls_features = self.rpn3d_cls_convs(cls_features)
            reg_features = self.rpn3d_bbox_convs(reg_features)
        box_preds = self.conv_box(reg_features)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        data_dict['reg_features'] = reg_features

        cls_preds = self.conv_cls(cls_features)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if not self.xyz_for_angles or not self.hwl_for_angles:
            # TODO: here we assume that for each class, there are only anchors with difference angles
            if self.xyz_for_angles:
                xyz_dim = self.num_anchors_per_location * 3
                xyz_shapes = (self.num_class, self.num_anchors_per_location // self.num_class, 3)
            else:
                xyz_dim = self.num_class * 3
                xyz_shapes = (self.num_class, 1, 3)
            if self.hwl_for_angles:
                hwl_dim = self.num_anchors_per_location * 3
                hwl_shapes = (self.num_class, self.num_anchors_per_location // self.num_class, 3)
            else:
                hwl_dim = self.num_class * 3
                hwl_shapes = (self.num_class, 1, 3)
            rot_dim = self.num_anchors_per_location * (self.box_coder.code_size - 6)
            rot_shapes = (self.num_class, self.num_anchors_per_location // self.num_class, (self.box_coder.code_size - 6))
            assert box_preds.shape[-1] == xyz_dim + hwl_dim + rot_dim
            xyz_preds, hwl_preds, rot_preds = torch.split(box_preds, [xyz_dim, hwl_dim, rot_dim], dim=-1)
            # anchors [Nz, Ny, Nx, N_cls*N_size=3*1, N_rot, 7]
            xyz_preds = xyz_preds.view(*xyz_preds.shape[:3], *xyz_shapes)
            hwl_preds = hwl_preds.view(*hwl_preds.shape[:3], *hwl_shapes)
            rot_preds = rot_preds.view(*rot_preds.shape[:3], *rot_shapes)
            # expand xyz and hwl
            if not self.xyz_for_angles:
                xyz_preds = xyz_preds.repeat(1, 1, 1, 1, rot_preds.shape[4] // xyz_preds.shape[4], 1)
            if not self.hwl_for_angles:
                hwl_preds = hwl_preds.repeat(1, 1, 1, 1, rot_preds.shape[4] // hwl_preds.shape[4], 1)
            box_preds = torch.cat([xyz_preds, hwl_preds, rot_preds], dim=-1)
            box_preds = box_preds.view(*box_preds.shape[:3], -1)

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if 'valids' in data_dict:
            self.forward_ret_dict['valids'] = data_dict['valids'].any(1)

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(cls_features)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training or 'gt_boxes' in data_dict:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            data_dict.update(targets_dict)
            data_dict['anchors'] = self.anchors
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            # TODO: check the code here, we add sigmoid in the generate predicted boxes, so set normalized to be True
            data_dict['cls_preds_normalized'] = False

        return data_dict
