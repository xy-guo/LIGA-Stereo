#include <torch/extension.h>

at::Tensor BuildCostVolume_forward_cuda(const at::Tensor &left,
                                        const at::Tensor &right,
                                        const at::Tensor &shift,
                                        const int downsample);

std::tuple<at::Tensor, at::Tensor> BuildCostVolume_backward_cuda(const at::Tensor &grad,
                                                                 const at::Tensor &shift,
                                                                 const int downsample);

// Interface for Python
at::Tensor BuildCostVolume_forward(const at::Tensor &left,
                                   const at::Tensor &right,
                                   const at::Tensor &shift,
                                   const int downsample)
{
  if (left.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildCostVolume_forward_cuda(left, right, shift, downsample);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::tuple<at::Tensor, at::Tensor> BuildCostVolume_backward(const at::Tensor &grad,
                                                            const at::Tensor &shift,
                                                            const int downsample)
{
  if (grad.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildCostVolume_backward_cuda(grad, shift, downsample);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("build_cost_volume_forward", &BuildCostVolume_forward, "BuildCostVolume_forward");
  m.def("build_cost_volume_backward", &BuildCostVolume_backward, "BuildCostVolume_backward");
}
