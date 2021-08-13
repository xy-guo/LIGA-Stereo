# Hourglass BEV backbone (same as DSGN. https://arxiv.org/abs/2001.03398)

import torch.nn as nn

from liga.models.backbones_3d_stereo.submodule import convbn, hourglass2d


class HgBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_channels = model_cfg.num_channels
        self.GN = model_cfg.GN

        self.rpn3d_conv2 = nn.Sequential(
            convbn(input_channels, self.num_channels, 3, 1, 1, 1, gn=self.GN),
            nn.ReLU(inplace=True))
        self.rpn3d_conv3 = hourglass2d(self.num_channels, gn=self.GN)
        self.num_bev_features = self.num_channels

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = self.rpn3d_conv2(spatial_features)

        data_dict['spatial_features_2d_prehg'] = x
        x = self.rpn3d_conv3(x, None, None)[0]
        data_dict['spatial_features_2d'] = x

        return data_dict
