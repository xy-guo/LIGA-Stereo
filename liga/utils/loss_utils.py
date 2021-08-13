import sys
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from liga.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu_differentiable
from . import box_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
            torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        assert torch.all((target == 1) | (target == 0)), 'labels should be 0 or 1 in focal loss.'
        assert input.shape == target.shape
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights  # .view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape == loss.shape[:-1]
            weights = weights.unsqueeze(-1)
            assert len(loss.shape) == len(weights.shape)
            loss = loss * weights

        return loss


class WeightedL2WithSigmaLoss(nn.Module):
    def __init__(self, code_weights: list = None):
        super(WeightedL2WithSigmaLoss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def l2_loss(diff, sigma=None):
        if sigma is None:
            loss = 0.5 * diff ** 2
        else:
            loss = 0.5 * (diff / torch.exp(sigma)) ** 2 + math.log(math.sqrt(6.28)) + sigma

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None, sigma: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights  # .view(1, 1, -1)

        loss = self.l2_loss(diff, sigma=sigma)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape == loss.shape[:-1]
            weights = weights.unsqueeze(-1)
            assert len(loss.shape) == len(weights.shape)
            loss = loss * weights

        return loss


class IOU3dLoss(nn.Module):
    def __init__(self):
        super(IOU3dLoss, self).__init__()

    @staticmethod
    def iou3d_loss(x, y):
        iou3d = boxes_iou3d_gpu_differentiable(x, y)
        return 1 - iou3d

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        input = input.contiguous()
        target = target.contiguous()
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        if input.size(0) > 0:
            loss = self.iou3d_loss(input, target)
        else:
            loss = (input - target).sum(1) * 0.

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape == loss.shape
            loss = loss * weights

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedBinaryCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, ...) float tensor.
                Predited logits for each class.
            target: (B, ...) float tensor.
                One-hot classification targets.
            weights: (B, ...) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        assert input.shape == target.shape
        assert input.shape == weights.shape
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none') * weights

        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


class InnerProductLoss(nn.Module):
    def __init__(self, code_weights: list = None):
        super(InnerProductLoss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()
        else:
            self.code_weights = None

    @staticmethod
    def ip_loss(product):
        return 1 - product.mean(dim=-1, keepdim=True)

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        product = input * target
        # code-wise weighting
        if self.code_weights is not None:
            product = product * self.code_weights  # .view(1, 1, -1)

        loss = self.ip_loss(product)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape == loss.shape[:-1]
            weights = weights.unsqueeze(-1)
            assert len(loss.shape) == len(weights.shape)
            loss = loss * weights

        return loss


class MergeLoss(nn.Module):
    def __init__(self, splits, multi_losses_cfg, code_weights):
        super(MergeLoss, self).__init__()
        self.multiple_losses = nn.ModuleList()
        self.splits = splits
        code_weights = np.array(code_weights)
        code_weights = np.split(code_weights, np.cumsum(splits)[:-1], 0)
        assert isinstance(code_weights, list)
        assert len(code_weights) == len(multi_losses_cfg)
        for cw, losses_cfg in zip(code_weights, multi_losses_cfg):
            reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
                else losses_cfg.REG_LOSS_TYPE
            self.multiple_losses.append(
                getattr(sys.modules[__name__], reg_loss_name)(code_weights=cw)
            )

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        inputs = torch.split(input, self.splits, -1)
        targets = torch.split(target, self.splits, -1)
        losses = []
        for input, target, reg_loss in zip(inputs, targets, self.multiple_losses):
            losses.append(reg_loss(input, target, weights))
        return losses
