# The backbone of our LIGA model.
# including 2D feature extraction, stereo volume construction, stereo network, stereo space -> 3D space conversion


import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from mmdet.models.builder import build_backbone, build_neck
from . import submodule
from .submodule import convbn_3d, feature_extraction_neck
from .cost_volume import BuildCostVolume


def project_pseudo_lidar_to_rectcam(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([-ys, -zs, xs], dim=-1)


def project_rectcam_to_pseudo_lidar(pts_3d):
    xs, ys, zs = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    return torch.stack([zs, -xs, -ys], dim=-1)


def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n, 1), device=pts_3d_rect.device)
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def unproject_image_to_rect(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    invP = torch.inverse(P4x4)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    return pts_3d[..., :3]


class LigaBackbone(nn.Module):
    def __init__(self, model_cfg, class_names, grid_size, voxel_size, point_cloud_range, boxes_gt_in_cam2_view=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # general config
        self.class_names = class_names
        self.GN = model_cfg.GN
        self.boxes_gt_in_cam2_view = boxes_gt_in_cam2_view
        self.fullres_stereo_feature = model_cfg.feature_neck.with_upconv

        # stereo config
        self.maxdisp = model_cfg.maxdisp
        self.downsample_disp = model_cfg.downsample_disp
        self.downsampled_depth_offset = model_cfg.downsampled_depth_offset
        self.num_hg = getattr(model_cfg, 'num_hg', 1)
        self.use_stereo_out_type = getattr(model_cfg, 'use_stereo_out_type', False)
        assert self.use_stereo_out_type in ["feature", "cost", "prob"]

        # volume construction config
        self.cat_img_feature = model_cfg.cat_img_feature
        self.img_feature_attentionbydisp = model_cfg.img_feature_attentionbydisp
        self.voxel_attentionbydisp = model_cfg.voxel_attentionbydisp
        self.rpn3d_dim = model_cfg.rpn3d_dim

        # volume config
        self.num_3dconvs = model_cfg.num_3dconvs
        self.cv_dim = model_cfg.cv_dim

        # feature extraction
        self.feature_backbone = build_backbone(model_cfg.feature_backbone)
        self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)
        if getattr(model_cfg, 'sem_neck', None):
            self.sem_neck = build_neck(model_cfg.sem_neck)
        else:
            self.sem_neck = None

        # cost volume
        self.build_cost = BuildCostVolume(model_cfg.cost_volume)

        # stereo network
        CV_INPUT_DIM = self.build_cost.get_dim(
            self.feature_neck.stereo_dim[-1])
        self.dres0 = nn.Sequential(
            convbn_3d(CV_INPUT_DIM, self.cv_dim, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(
            convbn_3d(self.cv_dim, self.cv_dim, 3, 1, 1, gn=self.GN))
        self.hg_stereo = nn.ModuleList()
        for _ in range(self.num_hg):
            self.hg_stereo.append(submodule.hourglass(self.cv_dim, gn=self.GN))

        # stereo predictions
        self.pred_stereo = nn.ModuleList()
        for _ in range(self.num_hg):
            self.pred_stereo.append(self.build_depth_pred_module())
        self.dispregression = submodule.disparityregression()

        # rpn3d convs
        RPN3D_INPUT_DIM = self.cv_dim if not (self.use_stereo_out_type != "feature") else 1
        if self.cat_img_feature:
            RPN3D_INPUT_DIM += self.feature_neck.sem_dim[-1]
        rpn3d_convs = []
        for i in range(self.num_3dconvs):
            rpn3d_convs.append(
                nn.Sequential(
                    convbn_3d(RPN3D_INPUT_DIM if i == 0 else self.rpn3d_dim,
                              self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                    nn.ReLU(inplace=True)))
        self.rpn3d_convs = nn.Sequential(*rpn3d_convs)
        self.rpn3d_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))
        self.num_3d_features = self.rpn3d_dim

        # prepare tensors
        self.prepare_depth(point_cloud_range, in_camera_view=False)
        self.prepare_coordinates_3d(point_cloud_range, voxel_size, grid_size)
        self.init_params()

        feature_backbone_pretrained = getattr(model_cfg, 'feature_backbone_pretrained', None)
        if feature_backbone_pretrained:
            self.feature_backbone.init_weights(pretrained=feature_backbone_pretrained)

    def build_depth_pred_module(self):
        return nn.Sequential(
            convbn_3d(self.cv_dim, self.cv_dim, 3, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.cv_dim, 1, 3, 1, 1, bias=False),
            nn.Upsample(scale_factor=self.downsample_disp, mode='trilinear', align_corners=True))

    def prepare_depth(self, point_cloud_range, in_camera_view=True):
        if in_camera_view:
            self.CV_DEPTH_MIN = point_cloud_range[2]
            self.CV_DEPTH_MAX = point_cloud_range[5]
        else:
            self.CV_DEPTH_MIN = point_cloud_range[0]
            self.CV_DEPTH_MAX = point_cloud_range[3]
        assert self.CV_DEPTH_MIN >= 0 and self.CV_DEPTH_MAX > self.CV_DEPTH_MIN
        depth_interval = (self.CV_DEPTH_MAX - self.CV_DEPTH_MIN) / self.maxdisp
        print('stereo volume depth range: {} -> {}, interval {}'.format(self.CV_DEPTH_MIN,
                                                                        self.CV_DEPTH_MAX, depth_interval))
        # prepare downsampled depth
        self.downsampled_depth = torch.zeros(
            (self.maxdisp // self.downsample_disp), dtype=torch.float32)
        for i in range(self.maxdisp // self.downsample_disp):
            self.downsampled_depth[i] = (
                i + self.downsampled_depth_offset) * self.downsample_disp * depth_interval + self.CV_DEPTH_MIN
        # prepare depth
        self.depth = torch.zeros((self.maxdisp), dtype=torch.float32)
        for i in range(self.maxdisp):
            self.depth[i] = (
                i + 0.5) * depth_interval + self.CV_DEPTH_MIN

    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def pred_depth(self, depth_conv_module, cost1, img_shape):
        cost1 = depth_conv_module(cost1)
        # cost1 = F.interpolate(
        #     cost1, [self.maxdisp, *img_shape],
        #     mode='trilinear',
        #     align_corners=True)
        cost1 = torch.squeeze(cost1, 1)
        cost1_softmax = F.softmax(cost1, dim=1)
        pred1 = self.dispregression(cost1_softmax,
                                    depth=self.depth.cuda())
        return cost1, cost1_softmax, pred1

    def get_local_depth(self, d_prob):
        with torch.no_grad():
            d = self.depth.cuda()[None, :, None, None]
            d_mul_p = d * d_prob
            local_window = 5
            p_local_sum = 0
            for off in range(0, local_window):
                cur_p = d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                p_local_sum += cur_p
            max_indices = p_local_sum.max(1, keepdim=True).indices
            pd_local_sum_for_max = 0
            for off in range(0, local_window):
                cur_pd = torch.gather(d_mul_p, 1, max_indices + off).squeeze(1)  # d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                pd_local_sum_for_max += cur_pd
            mean_d = pd_local_sum_for_max / torch.gather(p_local_sum, 1, max_indices).squeeze(1)
        return mean_d

    def forward(self, batch_dict):
        left = batch_dict['left_img']
        right = batch_dict['right_img']
        calib = batch_dict['calib']
        fu_mul_baseline = torch.as_tensor(
            [x.fu_mul_baseline for x in calib], dtype=torch.float32, device=left.device)
        if self.boxes_gt_in_cam2_view:
            calibs_Proj = torch.as_tensor(
                [x.K3x4 for x in calib], dtype=torch.float32, device=left.device)
        else:
            calibs_Proj = torch.as_tensor(
                [x.P2 for x in calib], dtype=torch.float32, device=left.device)

        N = batch_dict['batch_size']

        # feature extraction
        left_features = self.feature_backbone(left)
        left_features = [left] + list(left_features)
        right_features = self.feature_backbone(right)
        right_features = [right] + list(right_features)

        left_stereo_feat, left_sem_feat = self.feature_neck(left_features)
        right_stereo_feat, _ = self.feature_neck(right_features)

        if self.sem_neck is not None:
            batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
        else:
            batch_dict['sem_features'] = [left_sem_feat]

        batch_dict['rpn_feature'] = left_sem_feat

        # stereo matching: build stereo volume
        downsampled_depth = self.downsampled_depth.cuda()
        downsampled_disp = fu_mul_baseline[:, None] / \
            downsampled_depth[None, :] / (self.downsample_disp if not self.fullres_stereo_feature else 1)

        cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                                   None, None, downsampled_disp)

        # stereo matching network
        cost0 = self.dres0(cost_raw)
        cost0 = self.dres1(cost0) + cost0
        if len(self.hg_stereo) > 0:
            all_costs = []
            cur_cost = cost0
            for hg_stereo_module in self.hg_stereo:
                cost_residual, _, _ = hg_stereo_module(cur_cost, None, None)
                cur_cost = cur_cost + cost_residual
                all_costs.append(cur_cost)
        else:
            all_costs = [cost0]
        assert len(all_costs) > 0, 'at least one hourglass'

        # stereo matching: outputs
        batch_dict['depth_preds'] = []
        if not self.training:
            batch_dict['depth_preds_local'] = []
        batch_dict['depth_volumes'] = []
        batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
        for idx in range(len(all_costs)):
            upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[idx], all_costs[idx], left.shape[2:4])
            batch_dict['depth_volumes'].append(upcost_i)
            batch_dict['depth_preds'].append(pred_i)
            if not self.training:
                batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))

        # beginning of 3d detection part
        if self.use_stereo_out_type == "feature":
            out = all_costs[-1]
        elif self.use_stereo_out_type == "prob":
            out = cost_softmax_i.unsqueeze(1)
        elif self.use_stereo_out_type == "cost":
            out = upcost_i.unsqueeze(1)
        else:
            raise ValueError('wrong self.use_stereo_out_type option')
        out_prob = cost_softmax_i

        # convert plane-sweep into 3d volume
        coordinates_3d = self.coordinates_3d.cuda()
        batch_dict['coord'] = coordinates_3d
        norm_coord_imgs = []
        coord_imgs = []
        valids2d = []
        for i in range(N):
            c3d = coordinates_3d.view(-1, 3)
            if 'random_T' in batch_dict:
                random_T = batch_dict['random_T'][i]
                c3d = torch.matmul(c3d, random_T[:3, :3].T) + random_T[:3, 3]
            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            coord_img = project_rect_to_image(
                c3d,
                calibs_Proj[i].float().cuda())

            coord_img = torch.cat(
                [coord_img, c3d[..., 2:]], dim=-1)
            coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)

            coord_imgs.append(coord_img)

            img_shape = batch_dict['image_shape'][i]
            valid_mask_2d = (coord_img[..., 0] >= 0) & (coord_img[..., 0] <= img_shape[1]) & \
                (coord_img[..., 1] >= 0) & (coord_img[..., 1] <= img_shape[0])
            valids2d.append(valid_mask_2d)

            # TODO: crop augmentation
            crop_x1, crop_x2 = 0, left.shape[3]
            crop_y1, crop_y2 = 0, left.shape[2]
            norm_coord_img = (coord_img - torch.as_tensor([crop_x1, crop_y1, self.CV_DEPTH_MIN], device=coord_img.device)) / torch.as_tensor(
                [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1, self.CV_DEPTH_MAX - self.CV_DEPTH_MIN], device=coord_img.device)
            norm_coord_img = norm_coord_img * 2. - 1.
            norm_coord_imgs.append(norm_coord_img)
        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        coord_imgs = torch.stack(coord_imgs, dim=0)
        valids2d = torch.stack(valids2d, dim=0)

        batch_dict['norm_coord_imgs'] = norm_coord_imgs
        batch_dict['coord_imgs'] = coord_imgs

        valids = valids2d & (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
        batch_dict['valids'] = valids
        valids = valids.float()

        # Retrieve Voxel Feature from Cost Volume Feature
        Voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
        Voxel = Voxel * valids[:, None, :, :, :]

        if (self.voxel_attentionbydisp or
                (self.img_feature_attentionbydisp and self.cat_img_feature)):
            pred_disp = F.grid_sample(out_prob.detach()[:, None],
                                      norm_coord_imgs, align_corners=True)
            pred_disp = pred_disp * valids[:, None, :, :, :]

            if self.voxel_attentionbydisp:
                Voxel = Voxel * pred_disp

        # Retrieve Voxel Feature from 2D Img Feature
        if self.cat_img_feature:
            RPN_feature = left_sem_feat

            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            Voxel_2D = Voxel_2D * valids2d.float()[:, None, :, :, :]

            if self.img_feature_attentionbydisp:
                Voxel_2D = Voxel_2D * pred_disp

            if Voxel is not None:
                Voxel = torch.cat([Voxel, Voxel_2D], dim=1)
            else:
                Voxel = Voxel_2D

        # (64, 190, 20, 300)
        Voxel = self.rpn3d_convs(Voxel)  # (64, 190, 20, 300)
        batch_dict['volume_features_nopool'] = Voxel

        Voxel = self.rpn3d_pool(Voxel)  # [B, C, Nz, Ny, Nx] in cam view

        batch_dict['volume_features'] = Voxel

        return batch_dict
