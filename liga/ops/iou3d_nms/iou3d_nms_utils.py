"""
3D IoU Calculation and Rotated NMS
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
"""
import torch
import torch.autograd
from scipy.spatial import ConvexHull

from ...utils import common_utils
from . import numerical_jaccobian
from . import iou3d_nms_cuda, numerical_jaccobian
import numba


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou


def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


class BoxesIou3dDifferentiableFunction(torch.autograd.Function):
    @staticmethod
    def call_func(input):
        boxes_a, boxes_b = input
        assert boxes_a.shape[1] == boxes_b.shape[1] == 7

        # height overlap
        boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2)
        boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2)
        boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2)
        boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2)

        # # bev overlap
        # overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        # iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
        overlaps_bev = boxes_a.new_zeros([boxes_a.shape[0]])  # (N, M)
        iou3d_nms_cuda.boxes_overlap_bev_onebyone_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # 3d iou
        overlaps_3d = overlaps_bev * overlaps_h

        vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])
        vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])

        iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

        # print("iou", iou3d)
        return iou3d

    @staticmethod
    def forward(ctx, boxes_a, boxes_b):
        # boxes_a: [N, 7]
        # boxes_b: [M, 7]
        # return: [M, N]
        ctx.save_for_backward(boxes_a, boxes_b)
        return BoxesIou3dDifferentiableFunction.call_func((boxes_a, boxes_b))

    @staticmethod
    def backward(ctx, grad):
        # grad: [N]
        # grad_a [N, 7]
        boxes_a, boxes_b = ctx.saved_tensors
        fn = BoxesIou3dDifferentiableFunction.call_func
        grad_a = numerical_jaccobian.get_numerical_jacobian(fn, (boxes_a, boxes_b), boxes_a, eps=1e-3)
        # print("grad_out", grad)
        # print("grad_a", grad_a)
        grad_a = grad_a * grad[:, None]
        return grad_a, None


boxes_iou3d_gpu_differentiable = BoxesIou3dDifferentiableFunction.apply


if __name__ == "__main__":
    boxes_a = torch.rand([10, 7], dtype=torch.float32, device='cuda') * 6
    boxes_a.requires_grad = True
    boxes_b = boxes_a + torch.rand([10, 7], dtype=torch.float32, device='cuda') * 0.1

    # iou = boxes_iou3d_gpu_differentiable(boxes_a, boxes_b.detach())
    # iou1 = iou.clone()
    # print(iou)
    # iou.sum().backward()
    # g1 = boxes_a.grad.clone()
    # print(boxes_a.grad)
    # boxes_a.grad[...] = 0

    # iou = boxes_iou3d_gpu_differentiable(boxes_a, boxes_b.detach())
    # print(iou)
    # iou2 = iou.clone()
    # iou.sum().backward()
    # print(boxes_a.grad)
    # g2 = boxes_a.grad.clone()

    # print(iou1 - iou2)
    # print(g1 - g2)

    torch.autograd.gradcheck(boxes_iou3d_gpu_differentiable, (boxes_a, boxes_b.detach()), eps=1e-3, atol=0.001, rtol=1e-3)
