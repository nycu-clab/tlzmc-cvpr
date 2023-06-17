from typing import List
import torch
import torch.nn as nn
from util.rife import warp

def ResidualBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
    )


def DownsampleBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    )


def UpsampleBlock(in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.PReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
    )


def FeatBlock(in_channels, out_channels, stride=2):
    return torch.nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
        nn.PReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.PReLU()
    )


class ColumnBlock(nn.Module):
    def __init__(self, channels: List, down: bool) -> None:
        super(ColumnBlock, self).__init__()
        self.down = down

        if down:
            bridge = DownsampleBlock
        else:
            bridge = UpsampleBlock
            channels = channels[::-1]

        self.resblocks = nn.ModuleList([ResidualBlock(c, c, stride=1)
                                        for c in channels])
        self.bridge = nn.ModuleList([bridge(cin, cout)
                                     for cin, cout in zip(channels[:-1], channels[1:])])

    def forward(self, inputs) -> List:
        outputs = []

        if not self.down:
            inputs = inputs[::-1]

        for i, x in enumerate(inputs):
            out = self.resblocks[i](x)

            if i > 0:
                out += self.bridge[i - 1](outputs[-1])

            outputs.append(out)

        if not self.down:
            outputs = outputs[::-1]

        return outputs


class Backbone(nn.Module):
    def __init__(self, hidden_channels: List) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([FeatBlock(cin, cout, stride=1 if cin == 3 else 2)
                                       for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])

    def forward(self, x):
        feats = []
        for m in self.backbone:
            feats.append(m(x))
            x = feats[-1]

        return feats


class GridNet(nn.Module):  # GridNet([6, 64, 128, 192], [32, 64, 96], 6, 3)
    def __init__(self, in_channels: List, hidden_channels: List, columns, out_channels: int):
        super(GridNet, self).__init__()
        self.heads = nn.ModuleList([ResidualBlock(i, c, stride=1)
                                    for i, c in zip(in_channels, [hidden_channels[0]] + hidden_channels)])

        self.downs = nn.ModuleList([nn.Identity()])
        self.downs.extend([DownsampleBlock(cin, cout)
                           for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])

        columns -= 1  # minus 1 for heads
        self.columns = nn.Sequential(*[ColumnBlock(hidden_channels, n < columns // 2) for n in range(columns)])
        self.tail = ResidualBlock(hidden_channels[0], out_channels, stride=1)

    def forward(self, inputs):
        feats = []
        for i, x in enumerate(inputs):
            feat = self.heads[i](x)

            if i > 0:
                feat += self.downs[i - 1](feats[-1])

            feats.append(feat)

        feats.pop(0)  # head feat of image has been added into feat-1
        feats = self.columns(feats)
        output = self.tail(feats[0])

        return output, feats

import torch.nn.functional as F

class GridSynthNet3I(nn.Module):
    """
    use GridNet to synthesize the intermediate frame given two warped frame and features,
    which does not need to generate any mask.
    """

    def __init__(self):
        super(GridSynthNet3I, self).__init__()
        self.backbone = Backbone([3, 32, 64, 96])
        self.synth = GridNet([9, 96, 192, 288], [32, 64, 96], 6, 3)

    def forward(self, x0, x1, side, flow0, flow1):
        feats0 = self.backbone(x0)
        feats1 = self.backbone(x1)
        feats2 = self.backbone(side)
        warped_img0 = warp(x0, flow0)
        warped_img1 = warp(x1, flow1)
        flow = torch.cat([flow0, flow1], dim=1)

        warped_feats = [torch.cat([warped_img0, warped_img1, side], dim=1)]
        for level, (f0, f1, f2) in enumerate(zip(feats0, feats1, feats2)):
            s = 2 ** level
            flow_scaled = F.interpolate(flow, scale_factor=1. / s, mode="bilinear", align_corners=False) * 1. / s
            warped_f0 = warp(f0, flow_scaled[:, :2])
            warped_f1 = warp(f1, flow_scaled[:, 2:4])
            warped_feats.append(torch.cat([warped_f0, warped_f1, f2], dim=1))

        frame, _ = self.synth(warped_feats)

        return frame

from flownets import SPyNet
from modules import AugmentedNormalizedFlow, Conv2d, ConvTranspose2d


class AttMapSoftmax(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AttMapSoftmax, self).__init__()
        self.conv1_1_map = Conv2d(in_channels, 32, 3, stride=1)
        self.conv1_2_map = Conv2d(32, 32, 3, stride=1)
        self.conv1_3_map = Conv2d(64, 32, 3, stride=1)
        self.conv1_4_map = Conv2d(32, 32, 3, stride=1)

        self.conv2_1_map = Conv2d(32, 64, 3, stride=1)
        self.conv2_2_map = Conv2d(64, 64, 3, stride=1)
        self.conv2_3_map = Conv2d(128, 64, 3, stride=1)
        self.conv2_4_map = Conv2d(64, 64, 3, stride=1)

        self.conv3_1_map = Conv2d(64, 128, 3, stride=1)
        self.conv3_2_map = Conv2d(128, 128, 3, stride=1)
        self.conv3_3_map = Conv2d(128, 128, 3, stride=1)
        self.conv_final_map3 = Conv2d(32, out_channels, 1, stride=1)

        self.ds1 = nn.MaxPool2d(5, padding=2, stride=2)
        self.ds2 = nn.MaxPool2d(5, padding=2, stride=2)
        self.us1 = ConvTranspose2d(64, 32, 3, stride=2)
        self.us2 = ConvTranspose2d(128, 64, 3, stride=2)

        self.activ = torch.nn.Softmax(dim=1)


    def forward(self, x):
        map = nn.ReLU()(self.conv1_1_map(x))
        map_1 = nn.ReLU()(self.conv1_2_map(map))

        map = self.ds1(map_1)  # 128->128
        map = nn.ReLU()(self.conv2_1_map(map))  # 128->256
        map_2 = nn.ReLU()(self.conv2_2_map(map))  # 256->256

        map = self.ds2(map_2)  # 256->256
        map = nn.ReLU()(self.conv3_1_map(map))  # 256->512
        map = nn.ReLU()(self.conv3_2_map(map))  # 512
        map = nn.ReLU()(self.conv3_3_map(map))  # 512

        map = self.us2(map)  # 512->256
        map = nn.ReLU()(self.conv2_3_map(torch.cat((map_2, map), 1)))  # 512->256
        map = nn.ReLU()(self.conv2_4_map(map))  # 256->256

        map = self.us1(map)  # 256->128
        map = nn.ReLU()(self.conv1_3_map(torch.cat((map_1, map), 1)))  # 256->128
        map = nn.ReLU()(self.conv1_4_map(map))  # 128->128

        map = self.activ(self.conv_final_map3(map))  # 128->1

        return map

class GridSynthNetMod(nn.Module):
    """
    use GridNet to synthesize the intermediate frame given two warped frame and features,
    which does not need to generate any mask.
    """

    def __init__(self):
        super(GridSynthNetMod, self).__init__()
        self.backbone = Backbone([3, 32, 64, 96])
        self.synth_fin = GridNet([9, 96, 192, 288], [32, 64, 96], 6, 3)

    def forward(self, xsr, x0, x1, flow0, flow1):
        feats0 = self.backbone(x0)
        feats1 = self.backbone(x1)
        featssr = self.backbone(xsr)
        warped_img0 = warp(x0, flow0)
        warped_img1 = warp(x1, flow1)
        warped_feats = [torch.cat([warped_img0, xsr, warped_img1], dim=1)]
        flow = torch.cat([flow0, flow1], dim=1)
        for level, (f0, f1, fsr) in enumerate(zip(feats0, feats1, featssr)):

            s = 2 ** level
            flow_scaled = F.interpolate(flow, scale_factor=1. / s, mode="bilinear", align_corners=False) * 1. / s
            warped_f0 = warp(f0, flow_scaled[:, :2])
            warped_f1 = warp(f1, flow_scaled[:, 2:4])
            warped_feats.append(torch.cat([warped_f0, fsr, warped_f1], dim=1))

        frame, feats = self.synth_fin(warped_feats)

        return frame, feats


class GridSynthNet(nn.Module):
    """
    use GridNet to synthesize the intermediate frame given two warped frame and features,
    which does not need to generate any mask.
    """

    def __init__(self):
        super(GridSynthNet, self).__init__()
        self.backbone = Backbone([3, 32, 64, 96])
        self.synth = GridNet([6, 64, 128, 192], [32, 64, 96], 6, 3)

    def forward(self, x0, x1, flow0, flow1):
        feats0 = self.backbone(x0)
        feats1 = self.backbone(x1)
        warped_img0 = warp(x0, flow0)
        warped_img1 = warp(x1, flow1)
        flow = torch.cat([flow0, flow1], dim=1)

        warped_feats = [torch.cat([warped_img0, warped_img1], dim=1)]
        for level, (f0, f1) in enumerate(zip(feats0, feats1)):
            s = 2 ** level
            flow_scaled = F.interpolate(flow, scale_factor=1. / s, mode="bilinear", align_corners=False) * 1. / s
            warped_f0 = warp(f0, flow_scaled[:, :2])
            warped_f1 = warp(f1, flow_scaled[:, 2:4])
            warped_feats.append(torch.cat([warped_f0, warped_f1], dim=1))

        frame, feats = self.synth(warped_feats)

        return frame, feats

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, no_mask=False):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )

        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )

        self.no_mask = no_mask
        if no_mask:
            self.lastconv = nn.ConvTranspose2d(c, 4, 4, 2, 1)  # 5 - 4 for bi-directional flow, 1 for mask
        else:
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)  # 5 - 4 for bi-directional flow, 1 for mask

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)

        if flow != None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)

        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)

        if self.no_mask:
            flow = tmp * scale * 2
            return flow
        else:
            flow = tmp[:, :4] * scale * 2
            mask = tmp[:, 4:5]
            return flow, mask

class BiMotionPredict(nn.Module):

    def __init__(self):
        super(BiMotionPredict, self).__init__()
        self.block0 = IFBlock(6, c=240, no_mask=True)
        self.block1 = IFBlock(12 + 4, c=150, no_mask=True)
        self.block2 = IFBlock(12 + 4, c=90, no_mask=True)

    def forward(self, x0, x1, scale=[4, 2, 1]):
        img0 = x0
        img1 = x1

        flow_list = []
        flow = None
        stu = [self.block0, self.block1, self.block2]

        for i in range(len(scale)):
            if flow != None:
                inputs = torch.cat((img0, img1, warped_img0, warped_img1), 1)
                flow_d = stu[i](inputs, flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])

            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        f1, f2 = torch.split(flow_list[-1], [2, 2], dim=1)

        return f1, f2