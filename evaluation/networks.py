import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck
from generalizedivisivenorm import GeneralizedDivisiveNorm
from modules import AugmentedNormalizedFlow, Conv2d, ConvTranspose2d
from typing import List

class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()
        self.divisor = None
        self.num_bitstreams = 1

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def _cal_base_cdf(self):
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                m._cal_base_cdf()

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).sum() if len(aux_loss) else torch.zeros(1, device=next(self.parameters()).device)


class FactorizedCoder(CompressesModel):
    """FactorizedCoder"""

    def __init__(self, num_priors, quant_mode='noise'):
        super(FactorizedCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.entropy_bottleneck = EntropyBottleneck(
            num_priors, quant_mode=quant_mode)

        self.divisor = 16
        self.num_bitstreams = 1

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        ret = self.entropy_bottleneck.compress(features, return_sym=return_hat)

        if return_hat:
            y_hat, strings, shape = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, strings, shape
        else:
            return ret

    def decompress(self, strings, shape):
        y_hat = self.entropy_bottleneck.decompress(strings, shape)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        y_tilde, likelihoods = self.entropy_bottleneck(features)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, likelihoods


class HyperPriorCoder(FactorizedCoder):
    """HyperPrior Coder"""

    def __init__(self, num_condition, num_priors, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise', use_quant=False):
        super(HyperPriorCoder, self).__init__(
            num_priors, quant_mode=quant_mode)
        self.use_mean = use_mean
        self.use_abs = not self.use_mean or use_abs

        if use_quant:
            self.conditional_bottleneck = __CONDITIONS__[condition](use_mean=use_mean, use_quant=use_quant, quant_mode=quant_mode)
        else:
            self.conditional_bottleneck = __CONDITIONS__[condition](use_mean=use_mean, quant_mode=quant_mode)

        if use_context:
            self.conditional_bottleneck = ContextModel(
                num_condition, num_condition * 2, self.conditional_bottleneck)
        self.hyper_analysis = nn.Sequential()
        self.hyper_synthesis = nn.Sequential()

        self.divisor = 64
        self.num_bitstreams = 2

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shape):
        stream, side_stream = strings
        y_shape, z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, z_likelihood)


class GoogleAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn=False):
        super(GoogleAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )

class GoogleBaseAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, simplify_gdn=False):
        super(GoogleBaseAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, simplify=simplify_gdn),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class GoogleSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, simplify_gdn=False):
        super(GoogleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True, simplify=simplify_gdn),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class GoogleHyperScaleSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperScaleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=3, stride=1, parameterizer=None)
        )


class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=5):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=kernel_size, stride=2)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=5):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class GoogleHyperPriorCoder(HyperPriorCoder):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5,
                 use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)

        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features * self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, num_filters, num_hyperpriors)


class AugmentedNormalizedAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_features *
                   (2 if use_affine else 1), kernel_size, stride=2),
            nn.Identity(),
            use_affine=use_affine, transpose=False, distribution=distribution
        )


class AugmentedNormalizedSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedSynthesisTransform, self).__init__(
            nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_affine else 1), kernel_size, stride=2),
            use_affine=use_affine, transpose=True, distribution=distribution
        )

class DQ_ResBlock(nn.Sequential):
    def __init__(self, num_filters):
        super().__init__(
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(num_filters, num_filters, 3)
        )

    def forward(self, input):
        return super().forward(input) + input


class DeQuantizationModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(DeQuantizationModule, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, 3)
        self.resblock = nn.Sequential(
            *[DQ_ResBlock(num_filters) for _ in range(num_layers)])
        self.conv2 = Conv2d(num_filters, num_filters, 3)
        self.conv3 = Conv2d(num_filters, out_channels, 3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3


class AugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_QE=False, use_affine=True,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.use_QE = use_QE
        self.num_layers = num_layers
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        for i in range(num_layers):
            self.add_module('analysis' + str(i), AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and init_code != 'zeros',
                distribution=init_code))
            self.add_module('synthesis' + str(i), AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and i != num_layers - 1,
                distribution=init_code))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features * 2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features * self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            pass

        if self.use_QE:
            self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.DQ = None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, input, code=None, jac=None):
        for i in range(self.num_layers):
            _, code, jac = self['analysis' + str(i)](input, code, jac)

            if i < self.num_layers - 1:
                input, _, jac = self['synthesis' + str(i)](input, code, jac)

        return input, code, jac

    def decode(self, input, code=None, jac=None):
        for i in range(self.num_layers - 1, -1, -1):
            input, _, jac = self['synthesis' + str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers - 1)

            if i or jac is not None:
                _, code, jac = self['analysis' + str(i)](input, code, jac, rev=True)

        return input, code, jac

    def entropy_model(self, input, code, jac=False):

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)

        Y_error, _, jac = self['synthesis' + str(self.num_layers - 1)](
            input, y_tilde, jac, last_layer=True)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, code=None, return_hat=False):
        input, code, _ = self.encode(input, code, jac=None)

        hyperpriors = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        side_stream, h_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(h_hat)

        ret = self.conditional_bottleneck.compress(
            code, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, z_hat = ret

            x_hat = self.decode(None, z_hat, jac=None)[0]

            if self.use_QE:
                x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [z_hat.size(), h_hat.size()]
        else:
            stream = ret
            return [stream, side_stream], [code.size(), h_hat.size()]

    def decompress(self, strings, shapes):
        stream, side_stream = strings
        z_shape, h_shape = shapes

        h_hat = self.entropy_bottleneck.decompress(side_stream, h_shape)

        condition = self.hyper_synthesis(h_hat)

        z_hat = self.conditional_bottleneck.decompress(
            stream, z_shape, condition=condition)

        reconstructed = self.decode(None, z_hat, jac=None)[0]

        if self.use_QE:
            reconstructed = self.DQ(reconstructed)

        return reconstructed

    def forward(self, input, code=None, jac=None):
        jac = [] if jac else None

        input, code, jac = self.encode(input, code, jac)

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)

        Y_error, _, jac = self['synthesis' + str(self.num_layers - 1)](
            input, y_tilde, jac, last_layer=True)

        input, code, hyper_code = None, y_tilde, z_tilde

        input, code, jac = self.decode(input, code, jac)

        if self.use_QE:
            input = self.DQ(input)

        return input, (y_likelihood, z_likelihood), Y_error


class CondAugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """CondAugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,  # Note: out_channels is useless
                 init_code='gaussian', use_affine=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'
                 ):
        super(CondAugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers

        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        for i in range(num_layers):
            self.add_module('analysis' + str(i), AugmentedNormalizedAnalysisTransform(
                in_channels * 2, num_features, num_filters[i], kernel_size,
                use_affine=use_affine and init_code != 'zeros', distribution=init_code))
            self.add_module('synthesis' + str(i), AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[i], kernel_size,
                use_affine=use_affine and init_code != 'zeros', distribution=init_code))

        self.hyper_analysis = GoogleHyperAnalysisTransform(num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(num_features * 2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features * self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            pass


        self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)


    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, input, code=None, jac=None, xc=None):
        for i in range(self.num_layers):
            # Concat input with condition (MC frame)
            cond = xc
            cond_input = torch.cat([input, cond], dim=1)
            _, code, jac = self['analysis'+str(i)](cond_input, code, jac)

            if i < self.num_layers-1:
                input, _, jac = self['synthesis'+str(i)](input, code, jac)

        return input, code, jac

    def decode(self, input, code=None, jac=None, xc=None):
        for i in range(self.num_layers-1, -1, -1):
            input, _, jac = self['synthesis'+str(i)](input, code, jac, rev=True, last_layer=i == self.num_layers-1)

            if i or jac is not None:
                # Concat input with condition (MC frame)
                cond = xc
                cond_input = torch.cat([input, cond], dim=1)
                _, code, jac = self['analysis'+str(i)](cond_input, code, jac, rev=True)

        return input, code, jac

    def entropy_model(self, input, code):
        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)

        condition = self.hyper_synthesis(z_tilde)
        y_tilde, y_likelihood = self.conditional_bottleneck(code, condition=condition)

        return y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, xc=None, x2_back=None, return_hat=False):
        assert not (x2_back is None), ValueError
        assert not (xc is None), ValueError

        code = None
        jac = None
        input, features, jac = self.encode(
            input, code, jac, xc=xc)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            input = x2_back

            x_hat, code, jac = self.decode(
                input, y_hat, jac, xc=xc)

            x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shapes, xc=None, x2_back=None):
        assert not (x2_back is None), ValueError
        assert not (xc is None), ValueError

        jac = None

        stream, side_stream = strings
        y_shape, z_shape = shapes[0]

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        input = x2_back

        x_hat, code, jac = self.decode(
            input, y_hat, jac, xc=xc)

        reconstructed = self.DQ(x_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None,
                x2_back=None,
                xc=None
                ):
        assert not (x2_back is None), ValueError
        assert not (xc is None), ValueError

        jac = [] if jac else None
        input, code, jac = self.encode(input, code, jac, xc=xc)

        y_tilde, z_tilde, y_likelihood, z_likelihood = self.entropy_model(input, code)

        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](input, y_tilde, jac, last_layer=True,
                                                                   layer=self.num_layers - 1)

        input, code, hyper_code = x2_back, y_tilde, z_tilde

        input, code, jac = self.decode(input, code, jac, xc=xc)

        input = self.DQ(input)

        return input, (y_likelihood, z_likelihood), x_2


class Rounder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.round(input)  # (input - input.min()) / (input - input.min()).max()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return (input * grad_output).contiguous()


class Adaptor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, map, code):
        ctx.save_for_backward(input)
        input = input * map + code
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output.contiguous(), None, None


class CondAugmentedNormalizedFlowHyperPriorCoderPredPrior(CondAugmentedNormalizedFlowHyperPriorCoder):
    def __init__(self, in_channels_predprior=3, num_predprior_filters=128, **kwargs):
        super(CondAugmentedNormalizedFlowHyperPriorCoderPredPrior, self).__init__(**kwargs)
        print(kwargs)
        if num_predprior_filters is None:  # When not specifying, it will align to num_filters
            num_predprior_filters = kwargs['num_filters']

        if self.use_mean or "Mixture" in kwargs["condition"]:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs[
                                                          'num_features'] * self.conditional_bottleneck.condition_size,
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      )
            self.PA = nn.Sequential(
                nn.Conv2d((kwargs['num_features'] * self.conditional_bottleneck.condition_size) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'] * self.conditional_bottleneck.condition_size, 1)
            )
        else:
            self.pred_prior = GoogleAnalysisTransform(in_channels_predprior,
                                                      kwargs['num_features'],
                                                      num_predprior_filters,  # num_filters=64,
                                                      kwargs['kernel_size'],  # kernel_size=3,
                                                      )
            self.PA = nn.Sequential(
                nn.Conv2d(kwargs['num_features'] * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, kwargs['num_features'], 1)
            )

        self.map_adaptor = nn.Sequential(
            Conv2d(kwargs['num_features']*3, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, kwargs['num_features'], 3, stride=1),
            nn.Sigmoid(),
        )

        self.num_features = kwargs['num_features']

    def entropy_model(self, input, code, temporal_cond, r_map=None):

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(temporal_cond)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            code = code * r_map_update + inverse_code

        y_tilde, y_likelihood = self.conditional_bottleneck(code, condition=condition)

        if r_map is not None:
            y_tilde = Adaptor.apply(y_tilde, r_map_update.detach(), inverse_code.detach())

        # y_tilde = code # No quantize on z2

        return y_tilde, z_tilde, y_likelihood, z_likelihood, r_map

    def compress(self, input, xc=None, x2_back=None, temporal_cond=None, return_hat=False,
                 r_map=None):
        assert not (xc is None), ValueError
        assert not (temporal_cond is None), ValueError

        code = None
        jac = None
        input, features, jac = self.encode(input, code, jac, xc=xc)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(hyperpriors, return_sym=True)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))
        ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            features = features * r_map_update + inverse_code
            ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if return_hat:
            jac = None
            stream, y_hat = ret

            if r_map is not None:
                r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
                inverse_map = 1 - r_map_update
                inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
                y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

            input = x2_back

            x_hat, code, jac = self.decode(
                input, y_hat, jac, xc=xc)

            x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()], r_map
        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shapes, xc=None, x2_back=None, temporal_cond=None, r_map=None):
        assert not (xc is None), ValueError
        assert not (temporal_cond is None), ValueError

        jac = None

        stream, side_stream = strings
        y_shape, z_shape = shapes

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        # Decode
        input = x2_back

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

        x_hat, code, jac = self.decode(
            input, y_hat, jac, xc=xc)

        reconstructed = self.DQ(x_hat)

        return reconstructed

    def forward(self, input, code=None, jac=None,
                x2_back=None,
                xc=None,
                temporal_cond=None,
                r_map=None
                ):

        assert not (x2_back is None)
        assert not (xc is None)
        assert not (temporal_cond is None)

        jac = [] if jac else None

        input, code, jac = self.encode(
            input, code, jac, xc=xc)

        y_tilde, z_tilde, y_likelihood, z_likelihood, r_map = self.entropy_model(input, code, temporal_cond, r_map)

        x_2, _, jac = self['synthesis' + str(self.num_layers - 1)](input, y_tilde, jac, last_layer=True)

        input, code, hyper_code = x2_back, y_tilde, z_tilde

        # Decode
        input, code, jac = self.decode(input, code, jac, xc=xc)

        input = self.DQ(input)

        return input, (y_likelihood, z_likelihood), x_2, r_map

class EfficientAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=3):
        super(EfficientAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=kernel_size, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=kernel_size, stride=2)
        )


class EfficientSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=3):
        super(EfficientSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=kernel_size, stride=1)
        )



class TLZMCAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=3):
        super(TLZMCAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=kernel_size, stride=2)
        )


class TLZMCSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors, kernel_size=3):
        super(TLZMCSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=kernel_size, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


def ANFNorm(num_features, mode, inverse=False):
    if mode in ["standard", "simplify"]:
        return GeneralizedDivisiveNorm(num_features, inverse, simplify=mode == "simplify")
    elif mode == "layernorm":
        return nn.InstanceNorm2d(num_features)
    elif mode == "pass":
        return nn.Sequential()


class ANATransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(ANATransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features * (2 if use_code else 1), non_local=True) if use_attn else nn.Identity()
        )


class ANSTransformX(nn.Sequential):
    def __init__(self, out_channels, num_cond_frames, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False, gs=False):
        super(ANSTransformX, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2)
        )


class ANSTransform(nn.Module):
    def __init__(self, out_channels, num_cond_frames, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False, gs=False):
        super(ANSTransform, self).__init__()
        self.network = nn.Sequential(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2)
        )

    def forward(self, x):
        y = self.network(x)

        return y



class ANABaseTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False):
        super(ANABaseTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            ANFNorm(num_filters, mode=gdn_mode),
            Conv2d(num_filters, num_features *
                   (2 if use_code else 1), kernel_size, stride=2),
            AttentionBlock(num_features * (2 if use_code else 1), non_local=True) if use_attn else nn.Identity()
        )


class ANSBaseTransform(nn.Sequential):
    def __init__(self, out_channels, num_cond_frames, num_features, num_filters, kernel_size, use_code, distribution, gdn_mode, use_attn=False, integerlize=False, gs=False):
        super(ANSBaseTransform, self).__init__(
            AttentionBlock(
                num_features, non_local=True) if use_attn else nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            ANFNorm(num_filters, inverse=True, mode=gdn_mode),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_code else 1), kernel_size, stride=2)
        )


class GainUnit(nn.Module):

    def __init__(self, lmdas: List, vec_size: int, init="progress") -> None:
        super(GainUnit, self).__init__()
        num_vec = len(lmdas)
        unit = torch.empty((num_vec, vec_size))

        if init == "progress":
            for i in range(num_vec):
                unit[i] = 1.5 ** (-i)
        elif init == "one":
            unit[:] = 1
        else:
            raise ValueError(f"Don't support init={init}")

        self.register_buffer("lmdas", torch.tensor(lmdas, dtype=float))
        self.unit = nn.Parameter(unit, requires_grad=True)
        self.vec = 1

    def set_unit(self, lmda: int):
        # only works for batch = 1
        for i, v in enumerate(self.lmdas):
            if lmda >= v: break
        else:
            self.vec = self.unit[-1] * lmda / self.lmdas[-1]
            return

        if i == 0:
            self.vec = self.unit[0] * lmda / self.lmdas[0]
        else:
            l = (lmda - self.lmdas[i]) / (self.lmdas[i - 1] - self.lmdas[i])
            self.vec = torch.pow(self.unit[i - 1], l) * torch.pow(self.unit[i], 1 - l)

    def set_unit_idx(self, idx):
        self.vec = self.unit[idx]

    def forward(self, x, inverse):
        vec = self.vec.view(x.size(0), x.size(1), 1, 1)

        if not inverse:
            x = x * vec
        else:
            x = x / vec

        return x

class CANFEfficientCoder(HyperPriorCoder):
    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3,  kernel_size=5, pred_kernel_size=3,  num_layers=1, init_code='gaussian', share_wei=False,
                 hyper_filters=128, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise', use_quant=False,
                 num_cond_frames=None, # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 num_predprior_filters=128,
                 k2=3):
        super(CANFEfficientCoder, self).__init__(num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode, use_quant)

        self.num_layers = num_layers
        self.share_wei = share_wei

        self.num_predprior_filters = num_predprior_filters

        if num_cond_frames is None:
            num_cond_frames = in_channels
        if num_predprior_filters:
            num_pred_frames = num_cond_frames
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        # region - setup the encode and decode pairs
        for i in range(num_layers):
            ks = kernel_size if i == 0 else k2
            # print(f"{i} analysis")
            self.add_module('analysis' + str(i), ANATransform(
                in_channels + num_cond_frames, num_features, num_filters[i], ks,
                use_code=False and init_code != 'zeros',
                distribution='gaussian', gdn_mode='standard',
                use_attn=False and i == num_layers - 1))
            # print(f"{i} synthesis")
            self.add_module('synthesis' + str(i), ANSTransformX(
                in_channels, num_cond_frames, num_features, num_filters[i], ks,
                use_code=False and i != num_layers - 1, gs=False,
                distribution='gaussian', gdn_mode='standard', use_attn=False and i == num_layers - 1))

        self.hyper_analysis = EfficientAnalysisTransform(num_features, hyper_filters, num_hyperpriors, 3)
        self.hyper_synthesis = EfficientSynthesisTransform(num_features * self.conditional_bottleneck.condition_size,
                                                           hyper_filters, num_hyperpriors, 3)

        # to make this subscribable
        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)

        self.pred_prior = GoogleAnalysisTransform(num_pred_frames, num_features, num_predprior_filters,
                                                      pred_kernel_size, simplify_gdn=False)
        self.PA = nn.Sequential(
                nn.Conv2d(num_features * (1 + self.conditional_bottleneck.condition_size), 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, num_features * self.conditional_bottleneck.condition_size, 1)
            )

        self.map_adaptor = nn.Sequential(
            Conv2d(num_features*3, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, num_features, 3, stride=1),
            nn.Sigmoid(),
        )

        self.num_features = num_features

    def __getitem__(self, key):
        return self.__getattr__(key)

    def set_unit(self, lmda):
        self.gain_unit.set_unit(lmda)
        self.hyper_gain_unit.set_unit(lmda)

    def compress(self, x, xc=None, x2_back=None, temporal_cond=None, return_hat=False, code=None,
                 r_map=None):
        y = 0
        z = 0

        for i in range(self.num_layers):
            inputs = torch.cat([x, xc], dim=1) if isinstance(xc, torch.Tensor) else x
            y = y + self[f'analysis{i}'](inputs)

            if i < self.num_layers - 1:
                x = x - self[f'synthesis{i}'](y)

        z = z + self.hyper_analysis(y)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            z, return_sym=True)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        features = y
        hyperpriors = z

        ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            features = features * r_map_update + inverse_code

            ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret

            if r_map is not None:
                # y_hat = Adaptor.apply(y_hat, r_map.detach(), inverse_code.detach())

                r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
                print("Skipped samples %: ", (r_map_update.sum()) / (features.size(2) * features.size(3) * features.size(1)))
                #print("Efficiency %: ", len(stream_) / len(stream))
                inverse_map = 1 - r_map_update
                inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
                y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

            x_hat, _ = self.decode(xc, y_hat, self.num_layers, xc=xc)

            x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()], r_map

        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shapes, xc=None, x2_back=None, temporal_cond=None, r_map=None):
        assert not (xc is None)
        assert not (x2_back is None)
        assert not (temporal_cond is None)

        stream, side_stream = strings
        y_shape, z_shape = shapes

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

        x_hat, code = self.decode(
            xc, y_hat, self.num_layers, xc=xc)

        reconstructed = self.DQ(x_hat)

        return reconstructed

    def encode(self, x, y, z, layer, cond_input, r_map=None):
        for i in range(layer):
            inputs = torch.cat([x, cond_input], dim=1)
            y = y + self[f'analysis{i}'](inputs)

            if i < layer - 1:
                x = x - self[f'synthesis{i}'](y)

        z = z + self.hyper_analysis(y)
        z_tilde, z_likelihood = self.entropy_bottleneck(z)

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(cond_input)
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            y = y * r_map_update + inverse_code

        y_tilde, y_likelihood = self.conditional_bottleneck(y, condition=condition)

        if r_map is not None:
            y_tilde = Adaptor.apply(y_tilde, r_map_update.detach(), inverse_code.detach())

        x = x - self[f'synthesis{layer-1}'](y_tilde)

        return x, y_tilde, z_tilde, y_likelihood, z_likelihood, r_map

    def decode(self, x, y, layer, xc):
        for i in range(layer - 1, -1, -1):
            x = x + self[f'synthesis{i}'](y)

            if i:  #not use analysis0
                inputs = torch.cat([x, xc], dim=1)
                y = y - self[f'analysis{i}'](inputs)

        return x, y

    def forward(self, input, x2_back=None, xc=None, temporal_cond=None, r_map=None):
        assert not (x2_back is None), "x2_back should be specified"
        assert not (xc is None), "xc should be specified"

        x_2, y2_quant, z2_quant, y_likelihood, z_likelihood, r_map = self.encode(input, 0, 0, self.num_layers, xc, r_map=r_map)
        decoded_x, decoded_z = self.decode(x2_back, y2_quant, self.num_layers, xc)

        decoded_x = self.DQ(decoded_x)

        return decoded_x, (y_likelihood, z_likelihood), x_2, r_map

class TLZMCCANFCoder(HyperPriorCoder):
    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3,  kernel_size=5, pred_kernel_size=5,  num_layers=1, init_code='gaussian', share_wei=False,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise', use_quant=False,
                 num_cond_frames=None,  # Set 1 when only MC frame is for condition ; >1 whwn multi-refertence frames as conditions
                 num_predprior_filters=128,
                 k2=3):
        super(TLZMCCANFCoder, self).__init__(num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode, use_quant)

        self.num_layers = num_layers
        self.share_wei = share_wei

        self.num_predprior_filters = num_predprior_filters

        if num_cond_frames is None:
            num_cond_frames = in_channels
        if num_predprior_filters:
            num_pred_frames = num_cond_frames
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        # region - setup the encode and decode pairs
        for i in range(num_layers):
            ks = kernel_size
            # print(f"{i} analysis")
            self.add_module('analysis' + str(i), ANATransform(
                in_channels + num_cond_frames, num_features, num_filters[i], ks,
                use_code=False and init_code != 'zeros',
                distribution='gaussian', gdn_mode='standard',
                use_attn=False and i == num_layers - 1))
            # print(f"{i} synthesis")
            self.add_module('synthesis' + str(i), ANSTransformX(
                in_channels, num_cond_frames, num_features, num_filters[i], ks,
                use_code=False and i != num_layers - 1, gs=False,
                distribution='gaussian', gdn_mode='standard', use_attn=False and i == num_layers - 1))

        self.hyper_analysis = TLZMCAnalysisTransform(num_features, hyper_filters, num_hyperpriors, 3)
        self.hyper_synthesis = TLZMCSynthesisTransform(num_features * self.conditional_bottleneck.condition_size,
                                                           hyper_filters, num_hyperpriors, 3)

        # to make this subscribable
        for name, m in self.named_children():
            if "ana" in name or "syn" in name:
                m.name = name

        self.DQ = DeQuantizationModule(in_channels, in_channels, 64, 6)

        self.pred_prior = GoogleAnalysisTransform(num_pred_frames, num_features * self.conditional_bottleneck.condition_size, num_predprior_filters,
                                                      pred_kernel_size, simplify_gdn=False)
        self.PA = nn.Sequential(
                nn.Conv2d(num_features * (self.conditional_bottleneck.condition_size) * 2, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, 640, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(640, num_features * self.conditional_bottleneck.condition_size, 1)
            )

        self.map_adaptor = nn.Sequential(
            Conv2d(num_features*3, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            Conv2d(128, num_features, 3, stride=1),
            nn.Sigmoid(),
        )

        self.num_features = num_features

    def __getitem__(self, key):
        return self.__getattr__(key)

    def set_unit(self, lmda):
        self.gain_unit.set_unit(lmda)
        self.hyper_gain_unit.set_unit(lmda)

    def compress(self, x, xc=None, x2_back=None, temporal_cond=None, return_hat=False, code=None,
                 r_map=None):
        y = 0
        z = 0

        for i in range(self.num_layers):
            inputs = torch.cat([x, xc], dim=1) if isinstance(xc, torch.Tensor) else x
            y = y + self[f'analysis{i}'](inputs)

            if i < self.num_layers - 1:
                x = x - self[f'synthesis{i}'](y)

        z = z + self.hyper_analysis(y)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            z, return_sym=True)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        features = y
        hyperpriors = z

        ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            features = features * r_map_update + inverse_code

            ret = self.conditional_bottleneck.compress(features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret

            if r_map is not None:
                # y_hat = Adaptor.apply(y_hat, r_map.detach(), inverse_code.detach())

                r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
                print("Skipped samples %: ", (r_map_update.sum()) / (features.size(2) * features.size(3) * features.size(1)))
                #print("Efficiency %: ", len(stream_) / len(stream))
                inverse_map = 1 - r_map_update
                inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
                y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

            x_hat, _ = self.decode(xc, y_hat, self.num_layers, xc=xc)

            x_hat = self.DQ(x_hat)

            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()], r_map

        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shapes, xc=None, x2_back=None, temporal_cond=None, r_map=None):
        assert not (xc is None)
        assert not (x2_back is None)
        assert not (temporal_cond is None)

        stream, side_stream = strings
        y_shape, z_shape = shapes

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        hp_feat = self.hyper_synthesis(z_hat)
        pred_feat = self.pred_prior(temporal_cond)

        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            y_hat = Adaptor.apply(y_hat, r_map_update.detach(), inverse_code.detach())

        x_hat, code = self.decode(
            xc, y_hat, self.num_layers, xc=xc)

        reconstructed = self.DQ(x_hat)

        return reconstructed

    def encode(self, x, y, z, layer, cond_input, r_map=None):
        for i in range(layer):
            inputs = torch.cat([x, cond_input], dim=1)
            y = y + self[f'analysis{i}'](inputs)

            if i < layer - 1:
                x = x - self[f'synthesis{i}'](y)

        z = z + self.hyper_analysis(y)
        z_tilde, z_likelihood = self.entropy_bottleneck(z)

        hp_feat = self.hyper_synthesis(z_tilde)
        pred_feat = self.pred_prior(cond_input)
        condition = self.PA(torch.cat([hp_feat, pred_feat], dim=1))

        if r_map is not None:
            r_map_update = Rounder.apply(self.map_adaptor(torch.cat([r_map, condition], dim=1)))
            inverse_map = 1 - r_map_update
            inverse_code = inverse_map * condition[:, 0:self.num_features, :, :]
            y = y * r_map_update + inverse_code

        y_tilde, y_likelihood = self.conditional_bottleneck(y, condition=condition)

        if r_map is not None:
            y_tilde = Adaptor.apply(y_tilde, r_map_update.detach(), inverse_code.detach())

        x = x - self[f'synthesis{layer-1}'](y_tilde)

        return x, y_tilde, z_tilde, y_likelihood, z_likelihood, r_map

    def decode(self, x, y, layer, xc):
        for i in range(layer - 1, -1, -1):
            x = x + self[f'synthesis{i}'](y)

            if i:  #not use analysis0
                inputs = torch.cat([x, xc], dim=1)
                y = y - self[f'analysis{i}'](inputs)

        return x, y

    def forward(self, input, x2_back=None, xc=None, temporal_cond=None, r_map=None):
        assert not (x2_back is None), "x2_back should be specified"
        assert not (xc is None), "xc should be specified"

        x_2, y2_quant, z2_quant, y_likelihood, z_likelihood, r_map = self.encode(input, 0, 0, self.num_layers, xc, r_map=r_map)
        decoded_x, decoded_z = self.decode(x2_back, y2_quant, self.num_layers, xc)

        decoded_x = self.DQ(decoded_x)

        return decoded_x, (y_likelihood, z_likelihood), x_2, r_map

__CODER_TYPES__ = {
    "GoogleHyperPriorCoder": GoogleHyperPriorCoder,
    "ANFHyperPriorCoder": AugmentedNormalizedFlowHyperPriorCoder,
    "CondANFHyperPriorCoder": CondAugmentedNormalizedFlowHyperPriorCoder,
    "CondANFHyperPriorCoderPredPrior": CondAugmentedNormalizedFlowHyperPriorCoderPredPrior,
}