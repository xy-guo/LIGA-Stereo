# Anchor-based BEV detection head template. Mofidied from OpenPCDet. https://github.com/open-mmlab/OpenPCDet

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import copy

from liga.utils import box_coder_utils, common_utils, loss_utils
from liga.utils.common_utils import dist_reduce_mean
from liga.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class NormalizeLayer(nn.Module):
    def __init__(self, type, channel, momentum=0.99):
        super().__init__()
        self.channel = channel
        self.type = type
        self.momentum = momentum
        assert type in ['scale', 'cw_scale', "center+scale", "cw_center+scale"]

        self.channel_wise = False
        self.do_centering = False
        self.do_scaling = False
        self.scaling_method = "abs"

        if self.type == "scale":
            self.register_buffer("scale", torch.ones(1, 1))
            self.do_scaling = True
        elif self.type == "cw_scale":
            self.register_buffer("scale", torch.ones(1, channel))
            self.do_scaling = self.channel_wise = True
        elif self.type == "center+scale":
            self.register_buffer("center", torch.ones(1, 1))
            self.register_buffer("scale", torch.ones(1, 1))
            self.do_scaling = self.do_centering = True
        elif self.type == "cw_center+scale":
            self.register_buffer("center", torch.ones(1, channel))
            self.register_buffer("scale", torch.ones(1, channel))
            self.do_scaling = self.do_centering = self.channel_wise = True
        else:
            raise ValueError("invalid normalization type")
        assert self.do_scaling or self.do_centering, "at least one of scaling or centering normalization"

    def forward(self, inputs):
        if self.do_centering:
            x1 = inputs - self.center
        else:
            x1 = inputs
        if self.do_scaling:
            x2 = x1 / self.scale
        else:
            x2 = x1

        if self.training:
            self.update(inputs)
        return x2

    @torch.no_grad()
    def update(self, x):
        assert len(x.shape) == 2
        bsize = torch.tensor(x.shape[0], dtype=torch.long, device=x.device)
        dist.all_reduce(bsize, op=dist.ReduceOp.SUM)
        if bsize <= 10:
            return

        if self.do_centering:
            sum_x = torch.sum(x, dim=0, keepdim=True)
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            new_center = sum_x / torch.clamp(bsize, min=1)
            if not self.channel_wise:
                new_center = new_center.mean(dim=-1, keepdim=True)

            self.center *= self.momentum
            self.center += new_center * (1 - self.momentum)
            # if dist.get_rank() == 0:
            #     print("mean_center", self.center)
            #     print("avg_new_center", new_center.mean().item(), "avg_mean_center", self.center.mean().item())

            x = x - new_center

        if self.do_scaling:
            if self.scaling_method == "abs":
                sum_x = torch.sum(x.abs(), dim=0, keepdim=True)
                dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
                new_scale = sum_x / torch.clamp(bsize, min=1)
            elif self.scaling_method == "std":
                sum_x_sq = torch.sum(x ** 2, dim=0, keepdim=True)
                dist.all_reduce(sum_x_sq, op=dist.ReduceOp.SUM)
                new_scale = torch.sqrt(sum_x_sq / torch.clamp(bsize, min=1))
                # if using std, normalize to 0.667
                new_scale = new_scale
            else:
                raise ValueError('invalid scale method')
            if not self.channel_wise:
                new_scale = new_scale.mean(dim=-1, keepdim=True)

            self.scale *= self.momentum
            self.scale += new_scale * (1 - self.momentum)

            # if dist.get_rank() == 0:
            #     print("mean_scale", self.scale)
            #     print("avg_new_scale", new_scale.mean(), "avg_mean_scale", self.scale.mean())


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.clamp_value = model_cfg.get("CLAMP_VALUE", 10)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.reduce_avg_factor = getattr(self.model_cfg, 'reduce_avg_factor', True)
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.do_feature_imitation = model_cfg.do_feature_imitation
        if self.do_feature_imitation:
            cfgs = model_cfg.imitation_cfg if isinstance(model_cfg.imitation_cfg, list) else [model_cfg.imitation_cfg]
            self.imitation_configs = cfgs

            conv_imitation_layers = []
            self.norm_imitation = nn.ModuleDict()
            for cfg in cfgs:
                layers = []
                if cfg.layer == "conv2d":
                    layers.append(nn.Conv2d(
                        cfg.channel, cfg.channel, kernel_size=cfg.ksize, padding=cfg.ksize // 2, stride=1, groups=1
                    ))
                elif cfg.layer == "conv3d":
                    layers.append(nn.Conv3d(
                        cfg.channel, cfg.channel, kernel_size=cfg.ksize, padding=cfg.ksize // 2, stride=1, groups=1
                    ))
                else:
                    assert cfg.layer == "none", f"invalid layer type {cfg.layer}"
                if cfg.use_relu:
                    layers.append(nn.ReLU())
                    assert cfg.normalize is None

                if cfg.normalize is not None:
                    self.norm_imitation[cfg.stereo_feature_layer] = NormalizeLayer(cfg.normalize, cfg.channel)
                else:
                    self.norm_imitation[cfg.stereo_feature_layer] = nn.Identity()

                if len(layers) <= 1:
                    conv_imitation_layers.append(layers[0])
                else:
                    conv_imitation_layers.append(nn.Sequential(*layers))

            if len(cfgs) > 1:
                self.conv_imitation = nn.ModuleList(conv_imitation_layers)
            else:
                self.conv_imitation = conv_imitation_layers[0]

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg, model_cfg=None):
        if anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg if model_cfg is None else model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = losses_cfg.get('REG_LOSS_TYPE', 'WeightedSmoothL1Loss')
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'imitation_loss_func',
            getattr(loss_utils, losses_cfg.get('IMITATION_LOSS_TYPE', 'WeightedL2WithSigmaLoss'))()
        )
        self.add_module(
            'iou_loss_func',
            getattr(loss_utils, losses_cfg.get('IOU_LOSS_TYPE', 'IOU3dLoss'))()
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        if self.reduce_avg_factor:
            ori_pos_normalizer = pos_normalizer.clone().view(-1).mean()
            pos_normalizer = dist_reduce_mean(pos_normalizer.view(-1).mean())
            # if dist.get_rank() == 0:
            #     print('rpn cls normalizer {}->{}'.format(ori_pos_normalizer.item(), pos_normalizer.item()))

        cls_weights /= (pos_normalizer + self.clamp_value)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    # @staticmethod
    # def add_sin_difference(boxes1, boxes2, dim=6):
    #     assert dim != -1
    #     rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
    #     rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
    #     boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
    #     boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
    #     return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        negtives = box_cls_labels == 0

        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        if self.reduce_avg_factor:
            pos_normalizer = dist_reduce_mean(pos_normalizer.view(-1).mean())
        reg_weights /= torch.clamp(pos_normalizer, min=self.clamp_value)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        pos_inds = reg_weights > 0
        box_preds_sin, reg_targets_sin = self.box_coder.process_before_loss(anchors[pos_inds], box_preds[pos_inds], box_reg_targets[pos_inds])
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights[pos_inds])  # [N, M]
        assert not isinstance(loc_loss_src, list)

        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        iou_loss_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS.get('iou_weight', 0.)
        if iou_loss_weight > 0:
            decoded_box_preds = self.box_coder.decode_torch(box_preds[pos_inds], anchors[pos_inds])
            decoded_reg_targets = self.box_coder.decode_torch(box_reg_targets[pos_inds], anchors[pos_inds])
            iou_loss_src = self.iou_loss_func(decoded_box_preds, decoded_reg_targets, weights=reg_weights[pos_inds])
            iou_loss = iou_loss_src.sum() / batch_size
            iou_loss = iou_loss * iou_loss_weight
            box_loss += iou_loss
            tb_dict['rpn_loss_iou'] = iou_loss_src.sum().item()

            reduced_iou_loss = dist_reduce_mean(iou_loss)
            # if dist.get_rank() == 0:
            #     print('rpn reg iou:', reduced_iou_loss.item(), f"<{type(self.iou_loss_func)}>")

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            dir_weight_normalizer = weights.sum(-1, keepdim=True)
            if self.reduce_avg_factor:
                dir_weight_normalizer = dist_reduce_mean(dir_weight_normalizer.view(-1).mean())
            dir_weight_normalizer = torch.clamp(dir_weight_normalizer, min=self.clamp_value)
            weights /= dir_weight_normalizer

            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_imitation_reg_layer_loss(self, features_preds, features_targets, imitation_cfg):
        features_preds = features_preds.permute(0, *range(2, len(features_preds.shape)), 1)
        features_targets = features_targets.permute(0, *range(2, len(features_targets.shape)), 1)

        # features_preds = features_preds[:, :, :, :features_targets.shape[-1]]
        batch_size = int(features_preds.shape[0])

        # box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # positives = box_cls_labels > 0
        # positives = positives.view(*features_preds.shape[:-1], self.num_anchors_per_location)
        # positives = torch.any(positives, dim=-1)
        if imitation_cfg["mode"] == "inbox":
            anchors_xyz = self.anchors[0][:, :, :, 0, 0, :3].clone()
            gt_boxes = self.forward_ret_dict['gt_boxes'][..., :7].clone()
            anchors_xyz[..., 2] = 0
            gt_boxes[..., 2] = 0
            positives = points_in_boxes_gpu(anchors_xyz.view(anchors_xyz.shape[0], -1, 3), gt_boxes)
            positives = (positives >= 0).view(*anchors_xyz.shape[:3])
        elif imitation_cfg["mode"] == "full":
            positives = features_preds.new_ones(*features_preds.shape[:3])
            if dist.get_rank() == 0:
                print("using full imitation mask")
        else:
            raise ValueError("wrong imitation mode")

        if len(features_targets.shape) == 5:
            # 3d feature
            positives = positives.unsqueeze(1).repeat(1, features_targets.shape[1], 1, 1)
        else:
            assert len(features_targets.shape) == 4

        pre_pos_sum = positives.float().sum()
        positives = positives & torch.any(features_targets != 0, dim=-1)
        post_pos_sum = positives.float().sum()
        # if dist.get_rank() == 0:
        #     print(f"rpn after filtering all zero: {pre_pos_sum} -> {post_pos_sum}")

        reg_weights = positives.float()
        pos_normalizer = positives.sum().float()
        if self.reduce_avg_factor:
            pos_normalizer = dist_reduce_mean(pos_normalizer.mean())
        reg_weights /= torch.clamp(pos_normalizer, min=self.clamp_value)

        pos_inds = reg_weights > 0
        pos_feature_preds = features_preds[pos_inds]
        pos_feature_targets = self.norm_imitation[imitation_cfg.stereo_feature_layer](features_targets[pos_inds])
        imitation_loss_src = self.imitation_loss_func(pos_feature_preds,
                                                      pos_feature_targets,
                                                      weights=reg_weights[pos_inds])  # [N, M]
        imitation_loss_src = imitation_loss_src.mean(-1)

        imitation_loss = imitation_loss_src.sum() / batch_size
        imitation_loss = imitation_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['imitation_weight']

        tb_dict = {
            'rpn_loss_imitation': imitation_loss.item(),
        }

        if pos_inds.sum() > 0:
            rel_err = torch.median(torch.abs((pos_feature_preds - pos_feature_targets) / pos_feature_targets))
            rel_err_mean = torch.mean(torch.abs((pos_feature_preds - pos_feature_targets) / pos_feature_targets))
            tb_dict['rel_err_imitation_feature'] = rel_err.item()
        else:
            tb_dict['rel_err_imitation_feature'] = 0.

        return imitation_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        if 'imitation_features_pairs' in self.forward_ret_dict:
            npairs = len(self.forward_ret_dict["imitation_features_pairs"])
            for feature_pairs in self.forward_ret_dict["imitation_features_pairs"]:
                features_preds = feature_pairs['pred']
                features_targets = feature_pairs['gt']
                tag = ('_' + feature_pairs['stereo_feature_name']) if npairs > 1 else ''
                imitation_loss, tb_dict_imitation = self.get_imitation_reg_layer_loss(
                    features_preds=features_preds,
                    features_targets=features_targets,
                    imitation_cfg=feature_pairs['config'])
                tb_dict_imitation = {k + tag: v for k, v in tb_dict_imitation.items()}
                tb_dict.update(tb_dict_imitation)
                rpn_loss += imitation_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
