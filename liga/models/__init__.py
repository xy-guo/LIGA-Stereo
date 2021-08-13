from collections import namedtuple

import numpy as np
import torch

from .detectors_lidar import build_detector as build_lidar_detector
from .detectors_stereo import build_detector as build_stereo_detector


def build_network(model_cfg, num_class, dataset):
    if model_cfg['NAME'].startswith('stereo'):
        model = build_stereo_detector(
            model_cfg=model_cfg, num_class=num_class, dataset=dataset
        )
    else:
        model = build_lidar_detector(
            model_cfg=model_cfg, num_class=num_class, dataset=dataset
        )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'calib_ori', 'image_shape', 'gt_names']:
            continue
        if val.dtype in [np.float32, np.float64]:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        elif val.dtype in [np.uint8, np.int32, np.int64]:
            batch_dict[key] = torch.from_numpy(val).long().cuda()
        elif val.dtype in [bool]:
            pass
        else:
            raise ValueError(f"invalid data type {key}: {type(val)}")


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
