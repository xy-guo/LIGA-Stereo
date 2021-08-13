# modified from DSGN https://github.com/Jia-Research-Lab/DSGN
# sub-modules used in LIGA backbone.

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


def convbn(in_planes,
           out_planes,
           kernel_size,
           stride,
           pad,
           dilation=1,
           gn=False,
           groups=32):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=dilation if dilation > 1 else pad,
                  dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))


def convbn_3d(in_planes,
              out_planes,
              kernel_size,
              stride,
              pad,
              gn=False,
              groups=32):
    return nn.Sequential(
        nn.Conv3d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  padding=pad,
                  stride=stride,
                  bias=False),
        nn.BatchNorm3d(out_planes) if not gn else nn.GroupNorm(
            groups, out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride,
                 downsample,
                 pad,
                 dilation,
                 gn=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes, planes, 3, stride, pad, dilation, gn=gn),
            nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation, gn=gn)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()

    def forward(self, x, depth):
        assert len(x.shape) == 4
        assert len(depth.shape) == 1
        out = torch.sum(x * depth[None, :, None, None], 1)
        return out


class hourglass(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_3d(inplanes,
                      inplanes * 2,
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               stride=1,
                               pad=1,
                               gn=gn)

        self.conv3 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=2,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn_3d(inplanes * 2,
                      inplanes * 2,
                      kernel_size=3,
                      stride=1,
                      pad=1,
                      gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class hourglass2d(nn.Module):
    def __init__(self, inplanes, gn=False):
        super(hourglass2d, self).__init__()

        self.conv1 = nn.Sequential(
            convbn(inplanes,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv2 = convbn(inplanes * 2,
                            inplanes * 2,
                            kernel_size=3,
                            stride=1,
                            pad=1,
                            dilation=1,
                            gn=gn)

        self.conv3 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=2,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            convbn(inplanes * 2,
                   inplanes * 2,
                   kernel_size=3,
                   stride=1,
                   pad=1,
                   dilation=1,
                   gn=gn), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes * 2,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes *
                           2) if not gn else nn.GroupNorm(32, inplanes *
                                                          2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(inplanes * 2,
                               inplanes,
                               kernel_size=3,
                               padding=1,
                               output_padding=1,
                               stride=2,
                               bias=False),
            nn.BatchNorm2d(inplanes)
            if not gn else nn.GroupNorm(32, inplanes))  # +x

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu,
                          inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class upconv_module(nn.Module):
    def __init__(self, in_channels, up_channels):
        super(upconv_module, self).__init__()
        self.num_stage = len(in_channels) - 1
        self.conv = nn.ModuleList()
        self.redir = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            self.conv.append(
                convbn(in_channels[0] if stage_idx == 0 else up_channels[stage_idx - 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
            self.redir.append(
                convbn(in_channels[stage_idx + 1], up_channels[stage_idx], 3, 1, 1, 1)
            )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, feats):
        x = feats[0]
        for stage_idx in range(self.num_stage):
            x = self.conv[stage_idx](x)
            redir = self.redir[stage_idx](feats[stage_idx + 1])
            x = F.relu(self.up(x) + redir)
        return x


class feature_extraction_neck(nn.Module):
    def __init__(self, cfg):
        super(feature_extraction_neck, self).__init__()

        self.cfg = cfg
        self.in_dims = cfg.in_dims
        self.with_upconv = cfg.with_upconv
        self.start_level = cfg.start_level
        self.cat_img_feature = cfg.cat_img_feature

        self.sem_dim = cfg.sem_dim
        self.stereo_dim = cfg.stereo_dim
        self.spp_dim = getattr(cfg, 'spp_dim', 32)

        self.spp_branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(s, stride=s),
                convbn(self.in_dims[-1],
                       self.spp_dim,
                       1, 1, 0,
                       gn=cfg.GN,
                       groups=min(32, self.spp_dim)),
                nn.ReLU(inplace=True))
            for s in [(64, 64), (32, 32), (16, 16), (8, 8)]])

        concat_dim = self.spp_dim * len(self.spp_branches) + sum(self.in_dims[self.start_level:])

        if self.with_upconv:
            assert self.start_level == 2
            self.upconv_module = upconv_module([concat_dim, self.in_dims[1], self.in_dims[0]], [64, 32])
            stereo_dim = 32
        else:
            stereo_dim = concat_dim
            assert self.start_level >= 1

        self.lastconv = nn.Sequential(
            convbn(stereo_dim, self.stereo_dim[0], 3, 1, 1, gn=cfg.GN),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stereo_dim[0], self.stereo_dim[1],
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False))

        if self.cat_img_feature:
            self.rpnconv = nn.Sequential(
                convbn(concat_dim, self.sem_dim[0], 3, 1, 1, 1, gn=cfg.GN),
                nn.ReLU(inplace=True),
                convbn(self.sem_dim[0], self.sem_dim[1], 3, 1, 1, gn=cfg.GN),
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        feat_shape = tuple(feats[self.start_level].shape[2:])
        assert len(feats) == len(self.in_dims)

        spp_branches = []
        for branch_module in self.spp_branches:
            x = branch_module(feats[-1])
            x = F.interpolate(
                x, feat_shape,
                mode='bilinear',
                align_corners=True)
            spp_branches.append(x)

        concat_feature = torch.cat((*feats[self.start_level:], *spp_branches), 1)
        stereo_feature = concat_feature

        if self.with_upconv:
            stereo_feature = self.upconv_module([stereo_feature, feats[1], feats[0]])

        stereo_feature = self.lastconv(stereo_feature)

        if self.cat_img_feature:
            sem_feature = self.rpnconv(concat_feature)
        else:
            sem_feature = None

        return stereo_feature, sem_feature
