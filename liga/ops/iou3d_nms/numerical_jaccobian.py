import torch
from torch._six import container_abcs, istuple

from itertools import product


def iter_tensors(x, only_requiring_grad=False):
    if isinstance(x, torch.Tensor):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def get_numerical_jacobian(fn, input, target=None, eps=1e-3):
    """
    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """

    if target is None:
        target = input
    jacobian = torch.zeros_like(target)
    # print('jacobian', jacobian.shape)
    # import pdb
    # pdb.set_trace()

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [target]
    j_tensors = [jacobian]

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        assert not x_tensor.dtype.is_complex
        assert not x_tensor.is_sparse
        assert x_tensor.layout != torch._mkldnn

        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for idx in range(x_tensor.size(1)):
            orig = x_tensor[:, idx].clone()
            x_tensor[:, idx] = orig - eps
            outa = fn(input).clone()
            x_tensor[:, idx] = orig + eps
            outb = fn(input).clone()
            x_tensor[:, idx] = orig
            # print("outa, outb", idx, outa, outb)
            r = (outb - outa) / (2 * eps)
            d_tensor[:, idx] = r.detach()
    return jacobian
