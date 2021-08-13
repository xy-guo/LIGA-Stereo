import numpy as np
import torch

from liga.utils import box_utils, common_utils


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, div_by_diagonal=True, use_corners=False, use_tanh=False, tanh_range=3.14, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        self.div_by_diagonal = div_by_diagonal
        self.use_corners = use_corners
        self.use_tanh = use_tanh
        self.tanh_range = tanh_range
        if self.encode_angle_by_sincos:
            self.code_size += 1
        if self.use_corners:
            assert not encode_angle_by_sincos, "encode_angle_by_sincos should not be enabled when using corners"

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        if self.use_corners:
            return boxes
        anchors[..., 3:6] = torch.clamp_min(anchors[..., 3:6], min=1e-5)
        boxes[..., 3:6] = torch.clamp_min(boxes[..., 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        if self.div_by_diagonal:
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [common_utils.limit_period(rg - ra, offset=0.5, period=np.pi * 2)]
            # assert torch.all(rts[0] > -np.pi / 4 - 1e-4)
            # assert torch.all(rts[0] < np.pi / 4 + 1e-4)
        # print(rts[0] / 3.1415926 * 180)
        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def process_before_loss(self, anchors, pred, targets, dim=6):
        assert dim != -1

        if not self.use_corners:
            if self.use_tanh:
                pred[..., -1] = torch.tanh(pred[..., -1]) * (self.tanh_range / 2)
            boxes1, boxes2 = pred, targets
            rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
            rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
            boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
            boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
            return boxes1, boxes2
        else:
            pred = self.decode_torch(pred, anchors)
            pred_corners = box_utils.torch_boxes3d_to_corners3d_kitti_lidar(pred)
            target_corners = box_utils.torch_boxes3d_to_corners3d_kitti_lidar(targets)
            pred_corners = pred_corners.view(*pred_corners.shape[:-2], 24)
            target_corners = target_corners.view(*target_corners.shape[:-2], 24)
            return pred_corners, target_corners

    def decode_torch(self, box_encodings, anchors, decode_translation=True):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        if self.use_tanh:
            box_encodings[..., -1] = torch.tanh(box_encodings[..., -1]) * (self.tanh_range / 2)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not decode_translation:
            xa = 0
            ya = 0
            za = 0

        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        if self.div_by_diagonal:
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
