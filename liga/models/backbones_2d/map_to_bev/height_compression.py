# Convert sparse or dense 3D tensors into BEV 2D tensors by dimension rearrangement

import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.sparse_input = getattr(self.model_cfg, 'SPARSE_INPUT', True)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if self.sparse_input:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
            spatial_features = encoded_spconv_tensor.dense()
            batch_dict['volume_features'] = spatial_features
        else:
            spatial_features = batch_dict['volume_features']
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        if self.sparse_input:
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        else:
            batch_dict['spatial_features_stride'] = 1
        return batch_dict
