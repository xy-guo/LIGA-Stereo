import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from liga.ops.build_cost_volume import build_cost_volume_cuda


class _BuildCostVolume(Function):
    @staticmethod
    def forward(ctx, left, right, shift, downsample):
        ctx.save_for_backward(shift, )
        ctx.downsample = downsample
        assert torch.all(shift >= 0.)
        output = build_cost_volume_cuda.build_cost_volume_forward(
            left, right, shift, downsample)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        shift, = ctx.saved_tensors
        grad_left, grad_right = build_cost_volume_cuda.build_cost_volume_backward(
            grad_output, shift, ctx.downsample)
        return grad_left, grad_right, None, None


build_cost_volume = _BuildCostVolume.apply
