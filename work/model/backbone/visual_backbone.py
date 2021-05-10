# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):

        out = self.body(x)
        return_f = []
        return_f.append(out['3'])
        return_f.append(out['2'])
        return_f.append(out['1'])
        return_f.append(out['0'])

        return return_f


class Backbone(BackboneBase):

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrain_path: str):

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        backbone.load_state_dict(torch.load(pretrain_path))
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Neck(nn.Module):

    def __init__(self, n_levels=4, channels=[2048, 1024, 512, 256], fusion_size=32, lat_channels=128, args=None):
        super().__init__()
        self.n_levels = n_levels
        self.lat_conv = nn.ModuleList([nn.Conv2d(i, lat_channels,
                                                 kernel_size=(1, 1)) for i in channels])
        self.updown_conv = nn.ModuleList([nn.Conv2d(lat_channels, lat_channels,
                                                    kernel_size=(3, 3), stride=1, padding=1)
                                                for _ in range(n_levels-1)])
        self.fusion_size = fusion_size
        n = lat_channels * n_levels
        stride = 2 if args.stride else 1
        self.post_conv = nn.Sequential(
            nn.Conv2d(n, 1024, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),  # -> (64->32)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=(3, 3), padding=(1, 1), stride=(stride, stride)),  # -> (32->16)
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def upsample_add(self, feat1, feat2):
        _, _, H, W = feat2.size()
        return torch.nn.functional.interpolate(feat1, size=(H, W), mode='bilinear',
                                               align_corners=True) + feat2

    def forward(self, feats):
        assert len(feats) == self.n_levels
        for i in range(self.n_levels):
            feats[i] = self.lat_conv[i](feats[i])
        Out = []
        out = feats[0]
        out_append = torch.nn.functional.interpolate(out,
                                                     size=(self.fusion_size, self.fusion_size),
                                                     mode='bilinear',
                                                     align_corners=True)
        Out.append(out_append)
        for i in range(1, self.n_levels):
            out = self.updown_conv[i-1](self.upsample_add(out, feats[i]))
            out_append = torch.nn.functional.interpolate(out, size=(self.fusion_size, self.fusion_size),
                                                         mode='bilinear',
                                                         align_corners=True)
            Out.append(out_append)
        out = torch.cat(Out, dim=1)
        out = self.post_conv(out)
        return out


class VisualBackbone(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.cnn = Backbone(args.backbone, train_backbone=True,
                            return_interm_layers=True,
                            dilation=args.dilation,
                            pretrain_path=args.cnn_path)
        self.neck = Neck(4, [2048, 1024, 512, 256], args=args)

    def forward(self, img):
        return self.neck(self.cnn(img))


def build_visual_backbone(args):
    return VisualBackbone(args)