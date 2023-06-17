"""
This file defines the core research contribution
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import random
import warnings
from modules import Conv2d, ConvTranspose2d
from conditional_module import set_condition, conditional_warping

from util.sampler import Resampler
from util.rife_m import IFNet_m
from util.carn import CARN
from util.gridnet import GridSynthNetMod


from utils import BitStreamIO
from networks import AugmentedNormalizedFlowHyperPriorCoder, CANFEfficientCoder
from entropy_models import estimate_bpp
from flownets import SPyNet
from typing import List


# suppress warnings
warnings.filterwarnings("ignore")

# set some parameters
torch.cuda.memory_summary(device=None, abbreviated=False)
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# set up autograd
torch.autograd.set_detect_anomaly(True)


def padding_it(x, div=64):
    (m, c, w, h) = x.size()
    p1 = (div - (w % div)) % div
    p2 = (div - (h % div)) % div

    pad = nn.ZeroPad2d(padding=(0, p2, 0, p1))
    size = x.shape
    return pad(x), size


class MotionPredictor(nn.Module):
    def __init__(self):
        super(MotionPredictor, self).__init__()

        self.MINet = IFNet_m().cuda()  # Motion warping network
        self.Resampler = Resampler()

    def predict(self, x_c, x_f):
        flow, merged = self.MINet(torch.cat((x_c, x_f), 1), timestep=0.5, returnflow=True)
        flow_cx = flow[:, :2]
        flow_fx = flow[:, 2:4]

        return merged[2], flow_cx, flow_fx

class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1)
        )

    def forward(self, input):
        return input + super().forward(input)

class MergingMapsGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(MergingMapsGenerator, self).__init__()
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

        map = self.ds1(map_1)
        map = nn.ReLU()(self.conv2_1_map(map))
        map_2 = nn.ReLU()(self.conv2_2_map(map))

        map = self.ds2(map_2)
        map = nn.ReLU()(self.conv3_1_map(map))
        map = nn.ReLU()(self.conv3_2_map(map))
        map = nn.ReLU()(self.conv3_3_map(map))

        map = self.us2(map)
        map = nn.ReLU()(self.conv2_3_map(torch.cat((map_2, map), 1)))
        map = nn.ReLU()(self.conv2_4_map(map))

        map = self.us1(map)
        map = nn.ReLU()(self.conv1_3_map(torch.cat((map_1, map), 1)))
        map = nn.ReLU()(self.conv1_4_map(map))

        map = self.activ(self.conv_final_map3(map))

        return map

class MultiFrameMergeNet(nn.Module):
    def __init__(self):
        super(MultiFrameMergeNet, self).__init__()
        self.generate_merging_maps = MergingMapsGenerator(in_channels=9, out_channels=3)

    def forward(self, x_hat_sr, x_ref_1, x_ref_2):

        merging_maps = self.generate_merging_maps(torch.cat((x_hat_sr, x_ref_1, x_ref_2), 1))  # 128->1
        x_hat_sr_map = (x_hat_sr * torch.unsqueeze(merging_maps[:, 0, :, :], 1))
        x_ref_1_map = (x_ref_1 * torch.unsqueeze(merging_maps[:, 1, :, :], 1))
        x_ref_2_map = (x_ref_2 * torch.unsqueeze(merging_maps[:, 2, :, :], 1))

        out = x_hat_sr_map + x_ref_1_map + x_ref_2_map

        return out

class Refinement(nn.Module):
    """Refinement UNet"""

    def __init__(self, in_channels, num_filters, out_channels=3):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            ResidualBlock(num_filters)
        )
        self.l2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )
        self.l3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ResidualBlock(num_filters)
        )

        self.d3 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d2 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.d1 = nn.Sequential(
            ResidualBlock(num_filters),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 3, padding=1)
        )

    def forward(self, *input):
        if len(input) == 1:
            input = input[0]
        else:
            input = torch.cat(input, dim=1)

        conv1 = self.l1(input)
        conv2 = self.l2(conv1)
        conv3 = self.l3(conv2)

        deconv3 = self.d3(conv3)
        deconv2 = self.d2(deconv3 + conv2)
        deconv1 = self.d1(deconv2 + conv1)

        return deconv1

class ResidualCoder(nn.Module):
    def __init__(self):
        super(ResidualCoder, self).__init__()

        # Base and Hyperprior Residual Compressors
        self.base_compressor = CANFEfficientCoder(num_filters=128,
                                                         num_features=128,
                                                         num_hyperpriors=128,
                                                         kernel_size=5, num_layers=2,
                                                         use_mean=True,
                                                         use_context=False,
                                                         quant_mode="RUN")
        self.enhancement_compressor = CANFEfficientCoder(num_filters=128,
                                                         num_features=128,
                                                         num_hyperpriors=128,
                                                         kernel_size=5, num_layers=2,
                                                         use_mean=True,
                                                         use_context=False,
                                                         quant_mode="RUN")


        self.skip_mask_generator = nn.Sequential(
            Conv2d(7, 32, 5, stride=2),
            ResidualBlock(32),
            Conv2d(32, 64, 5, stride=2),
            ResidualBlock(64),
            Conv2d(64, 128, 5, stride=2),
            ResidualBlock(128),
            Conv2d(128, 128, 5, stride=2),
            nn.Sigmoid()
        )

        # Down-Sampling Layers
        self.ds2 = nn.Sequential(
            nn.MaxPool2d(2)
        )

        # Super-Resolution Model
        self.sr = CARN(multi_scale=True, group=4)

        # Merging
        self.merging = MultiFrameMergeNet()

        # Refinement
        self.refine = Refinement(9, 64, 3)

        # Resampler Layer
        self.Resampler = Resampler()

    def compress(self, x_c, x_f, x_rife, x_t, flow_cx, flow_fx):
        with torch.no_grad():
            # Prepare inputs
            x_bar_sm = self.ds2(x_rife)
            x_t_sm = self.ds2(x_t)

            x_t_sm, size = padding_it(x_t_sm)
            x_bar_sm, size = padding_it(x_bar_sm)

            # Compress base layer
            _, res_strings_sm, res_shape_sm, _ = self.base_compressor.compress(x_t_sm,
                                                                                        xc=x_bar_sm,
                                                                                        x2_back=x_bar_sm,
                                                                                        temporal_cond=x_bar_sm,
                                                                                        return_hat=True)
            # Decompress base layer
            x_hat_sm = self.base_compressor.decompress(res_strings_sm, res_shape_sm,
                                                                xc=x_bar_sm,
                                                                x2_back=x_bar_sm,
                                                                temporal_cond=x_bar_sm)
            x_hat_sm = x_hat_sm[:, :, :size[2], :size[3]]
            x_sr = self.sr(x_hat_sm, 2)

            x_bar_mc_c = self.Resampler(x_c, flow_cx)
            x_bar_mc_f = self.Resampler(x_f, flow_fx)

            # Merging
            x_merge = self.merging(x_sr, x_bar_mc_c, x_bar_mc_f)

            # Refine
            x_prime = self.refine(torch.cat([x_c, x_merge, x_f], dim=1))

            x_t, size = padding_it(x_t)
            x_prime, size = padding_it(x_prime)
            flow_cx, size = padding_it(flow_cx)
            flow_fx, size = padding_it(flow_fx)

            # Skip map generation
            r_map = self.skip_mask_generator(torch.cat((flow_cx, x_prime, flow_fx), 1))

            # Compress enhancement layer
            _, res_strings, res_shape, _ = self.enhancement_compressor.compress(x_t,
                                                                               xc=x_prime,
                                                                               x2_back=x_prime,
                                                                               temporal_cond=x_prime,
                                                                               r_map=r_map,
                                                                               return_hat=True)
            # Decompress enhancement layer
            x_hat = self.enhancement_compressor.decompress(res_strings, res_shape,
                                                          xc=x_prime,
                                                          x2_back=x_prime,
                                                          temporal_cond=x_prime,
                                                          r_map=r_map)

            x_hat = x_hat[:, :, :size[2], :size[3]]

        return x_hat, res_strings_sm, res_shape_sm, res_strings, res_shape

    def decompress(self, x_c, x_f, x_rife, flow_cx, flow_fx, res_strings_sm, res_shape_sm, res_strings, res_shape):
        with torch.no_grad():
            # Decompress base layer
            x_bar_sm = self.ds2(x_rife)
            x_bar_sm, size = padding_it(x_bar_sm)
            x_hat_sm = self.base_compressor.decompress(res_strings_sm, res_shape_sm, xc=x_bar_sm,
                                                                x2_back=x_bar_sm, temporal_cond=x_bar_sm)
            x_hat_sm = x_hat_sm[:, :, :size[2], :size[3]]
            x_sr = self.sr(x_hat_sm, 2)

            x_bar_mc_c = self.Resampler(x_c, flow_cx)
            x_bar_mc_f = self.Resampler(x_f, flow_fx)

            # Merging
            x_merge = self.merging(x_sr, x_bar_mc_c, x_bar_mc_f)

            # Refine
            x_prime = self.refine(torch.cat([x_c, x_merge, x_f], dim=1))

            # Apply padding
            x_prime, size = padding_it(x_prime)
            flow_cx, size = padding_it(flow_cx)
            flow_fx, size = padding_it(flow_fx)

            # Compute skip map
            r_map = self.skip_mask_generator(torch.cat((flow_cx, x_prime, flow_fx), 1))

            # Decompress enhancement layer
            x_hat = self.enhancement_compressor.decompress(res_strings, res_shape, xc=x_prime, x2_back=x_prime,
                                                          temporal_cond=x_prime, r_map=r_map)
            x_hat = x_hat[:, :, :size[2], :size[3]]

        return x_hat



class Compressor(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        with torch.no_grad():
            self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_QE=True,
                                                                   use_affine=False,
                                                                   use_context=True, condition='GaussianMixtureModel',
                                                                   quant_mode='round').cuda()

        self.mp = MotionPredictor()

        self.res = ResidualCoder()

        conditional_warping(self.res.base_compressor, discrete=False, conditions=3, ver=2)
        conditional_warping(self.res.enhancement_compressor, discrete=False, conditions=3, ver=2)

    def get_condition(self, level=0):
        if level == 0:
            return torch.tensor([[0, 1, 0]], dtype=torch.float).cuda()
        elif level == 1:
            return torch.tensor([[1, 0, 0]], dtype=torch.float).cuda()


    def intra_coding(self, x):
        with torch.no_grad():
            x, size = padding_it(x)

            x_hat, likelihoods, _ = self.if_model(x)
            x_hat = x_hat[:, :, :size[2], :size[3]]
            x_hat = torch.clamp(x_hat, 0, 1)
            rate = estimate_bpp(likelihoods, input=x_hat).mean().item()
        return x_hat, rate

    def inter_coding(self, x_c, x_t, x_f, reference=False):
        self.eval()
        height, width = x_c.size()[2:]
        with torch.no_grad():

            x_f, size = padding_it(x_f)
            x_c, size = padding_it(x_c)
            x_t, size = padding_it(x_t)

            if reference:
                cond = self.get_condition(0)
            else:
                cond = self.get_condition(1)

            set_condition(self.res.base_compressor, cond)
            set_condition(self.res.enhancement_compressor, cond)  # non-ref

            x_bar_rife, flow_cx, flow_fx = self.mp.predict(x_c, x_f)

            _, strings_l, shapes_l, strings_h, shapes_h = self.res.compress(x_c, x_f, x_bar_rife, x_t, flow_cx, flow_fx)

            # write compressed data to files
            with BitStreamIO("residual_bits_base.bpp", 'w') as fp:
                fp.write(strings_l[0] + strings_l[1],
                         [shapes_l[0]] + [shapes_l[1]] + [torch.Size([1, 1, 1, 1])] + [torch.Size([1, 1, 1, 1])])
            # calculate compression rate
            size_byte = os.path.getsize("residual_bits_base.bpp")
            base_rate = size_byte * 8 / height / width

            # write compressed data to files
            with BitStreamIO("residual_bits_enhancement.bpp", 'w') as fp:
                fp.write(strings_h[0] + strings_h[1],
                         [shapes_h[0]] + [shapes_h[1]] + [torch.Size([1, 1, 1, 1])] + [torch.Size([1, 1, 1, 1])])
            # calculate compression rate
            size_byte = os.path.getsize("residual_bits_enhancement.bpp")
            enhancement_rate = size_byte * 8 / height / width

            print("Read from files... Real decoding")

            # read compressed data from file for base layer
            file_reader = BitStreamIO("residual_bits_base.bpp", "r")
            dec_strings_base_list = []
            dec_shapes_base_list = []
            for _ in range(4):
                # read each string and shape
                read_strings_base, read_shapes_base = file_reader.read()
                dec_strings_base_list.append(read_strings_base)
                dec_shapes_base_list.append(torch.Size(read_shapes_base[0]))

            # organize string and shape into nested list format
            dec_strings_base = [[dec_strings_base_list[0][0], dec_strings_base_list[1][0]],
                                [dec_strings_base_list[2][0], dec_strings_base_list[3][0]]]
            dec_shapes_base = [dec_shapes_base_list[0], dec_shapes_base_list[1]]

            # read compressed data from file for enhancement layer
            file_reader = BitStreamIO("residual_bits_enhancement.bpp", "r")
            dec_strings_enhancement_list = []
            dec_shapes_enhancement_list = []
            for _ in range(4):
                # read each string and shape
                read_strings_h, read_shapes_h = file_reader.read()
                dec_strings_enhancement_list.append(read_strings_h)
                dec_shapes_enhancement_list.append(torch.Size(read_shapes_h[0]))

            # organize string and shape into nested list format
            dec_strings_h = [[dec_strings_enhancement_list[0][0], dec_strings_enhancement_list[1][0]],
                             [dec_strings_enhancement_list[2][0], dec_strings_enhancement_list[3][0]]]
            dec_shapes_h = [dec_shapes_enhancement_list[0], dec_shapes_enhancement_list[1]]

            x_bar_rife, flow_cx, flow_fx = self.mp.predict(x_c, x_f)

            x_hat = self.res.decompress(x_c, x_f, x_bar_rife, flow_cx, flow_fx, dec_strings_base, dec_shapes_base, dec_strings_h,
                                        dec_shapes_h)

            x_hat = x_hat[:, :, :size[2], :size[3]]

        return x_hat, base_rate, enhancement_rate





