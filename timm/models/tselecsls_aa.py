"""PyTorch SelecSLS Net example for ImageNet Classification
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)
Author: Dushyant Mehta (@mehtadushy)

SelecSLS (core) Network Architecture as proposed in "XNect: Real-time Multi-person 3D
Human Pose Estimation with a Single RGB Camera, Mehta et al."
https://arxiv.org/abs/1907.00837

Based on ResNet implementation in https://github.com/rwightman/pytorch-image-models
and SelecSLS Net implementation in https://github.com/mehtadushy/SelecSLS-Pytorch
"""
import math

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model
from .helpers import load_pretrained
from .layers import SelectAdaptivePool2d, AntiAliasDownsampleLayer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from inplace_abn import InPlaceABN
    has_iabn = True
except ImportError:
    has_iabn = False

__all__ = ['TSelecSLSAA']  # model_registry will add each entrypoint fn to this

class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class FastSEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(FastSEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (3, 3),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'aa_tselecsls42': _cfg(
        url='',
        interpolation='bicubic'),
    'aa_tselecsls42b': _cfg(
        url='',
        interpolation='bicubic'),
    'aa_tselecsls60': _cfg(
        url='',
        interpolation='bicubic'),
    'aa_tselecsls60b': _cfg(
        url='',
        interpolation='bicubic'),
}


def conv_abn(in_chs, out_chs, k=3, stride=1, padding=None, dilation=1, activation="leaky_relu", activation_param=1e-3):
    if padding is None:
        padding = ((stride - 1) + dilation * (k - 1)) // 2
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, k, stride, padding=padding, dilation=dilation, bias=False),
        InPlaceABN(num_features=out_chs, activation=activation, activation_param=activation_param)
    )


class SelecSLSBlock(nn.Module):
    def __init__(self, in_chs, skip_chs, mid_chs, out_chs, is_first, stride,  use_se=True, anti_alias_layer=None):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.is_first = is_first
        assert stride in [1, 2]

        # Process input with 4 conv blocks with the same number of input and output channels
        if stride == 1:
            self.conv1 = conv_abn(in_chs, mid_chs, 3, 1 )
        else:
            if anti_alias_layer is None:
                self.conv1 = conv_abn(in_chs, mid_chs, 3, stride)
            else:
                self.conv1 = nn.Sequential(conv_abn(in_chs, mid_chs, 3, 1),
                        anti_alias_layer(channels=mid_chs, filt_size=3, stride=stride))
        self.conv2 = conv_abn(mid_chs, mid_chs, 1)
        self.conv3 = conv_abn(mid_chs, mid_chs // 2, 3)
        self.conv4 = conv_abn(mid_chs // 2, mid_chs, 1)
        self.conv5 = conv_abn(mid_chs, mid_chs // 2, 3)
        self.conv6 = conv_abn(2 * mid_chs + (0 if is_first else skip_chs), out_chs, 1, activation='identity')
        reduce_layer_planes = max(mid_chs // 4, 64)
        self.se = FastSEModule(mid_chs, reduce_layer_planes) if use_se else None
        self.relu = nn.ReLU()

    def forward(self, x):
        assert isinstance(x, list)
        assert len(x) in [1, 2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.se is not None:
            d4 = self.se( F.leaky_relu(-1000 *torch.cat([d2, d3], 1), negative_slope=1e-3) )
        else:
            d4 = torch.cat([d2, d3], 1)
        if self.is_first:
            out = self.relu(self.conv6(torch.cat([d1, d4], 1)))
            return [out, out]
        else:
            return [self.relu(self.conv6(torch.cat([d1, d4, x[1]], 1))), x[1]]


class TSelecSLSAA(nn.Module):
    """SelecSLS42 / SelecSLS60 / SelecSLS84

    Parameters
    ----------
    cfg : network config dictionary specifying block type, feature, and head args
    num_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    """

    if not has_iabn:
        raise " For TSelecSLS models, please install InplaceABN: 'pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11' "
    def __init__(self, cfg, num_classes=1000, in_chans=3, drop_rate=0.0, remove_aa_jit=False, global_pool='avg'):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TSelecSLSAA, self).__init__()

        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        #self.stem = conv_abn(in_chans*16, 64, stride=1)
        self.stem = conv_abn(in_chans, 32, stride=2)
        self.features = nn.Sequential(*[cfg['block'](*[b for a in [block_args,[anti_alias_layer]] for b in a]) for block_args in cfg['features']])
        self.head = nn.Sequential(*[conv_abn(*conv_args) for conv_args in cfg['head']])
        self.num_features = cfg['num_features']

        #self.global_pool = FastGlobalAvgPool2d()
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.fc = nn.Linear(self.num_features, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        #self.global_pool = FastGlobalAvgPool2d()
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.num_classes = num_classes
        del self.fc
        if num_classes:
            self.fc = nn.Linear(self.num_features, num_classes)
        else:
            self.fc = None

    def forward_features(self, x):
        #x = self.stem(self.space_to_depth(x))
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x).flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        return x


def _create_model(variant, pretrained, model_kwargs):
    cfg = {}
    if variant.startswith('aa_tselecsls42'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2, True),
            (64, 64, 64, 128, False, 1, True),
            (128, 0, 144, 144, True, 2, True),
            (144, 144, 144, 288, False, 1, True),
            (288, 0, 304, 304, True, 2, False),
            (304, 304, 304, 480, False, 1, False),
        ]
        # Head can be replaced with alternative configurations depending on the problem
        if variant == 'aa_tselecsls42b':
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (480, 960, 3, 2),
                (960, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            cfg['num_features'] = 1280
    elif variant.startswith('aa_tselecsls60'):
        cfg['block'] = SelecSLSBlock
        # Define configuration of the network after the initial neck
        cfg['features'] = [
            # in_chs, skip_chs, mid_chs, out_chs, is_first, stride
            (32, 0, 64, 64, True, 2, True),
            (64, 64, 64, 128, False, 1, True),
            (128, 0, 128, 128, True, 2, True),
            (128, 128, 128, 128, False, 1, True),
            (128, 128, 128, 288, False, 1, True),
            (288, 0, 288, 288, True, 2, False),
            (288, 288, 288, 288, False, 1, False),
            (288, 288, 288, 288, False, 1, False),
            (288, 288, 288, 416, False, 1, False),
        ]
        # Head can be replaced with alternative configurations depending on the problem
        if variant == 'aa_tselecsls60b':
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1280, 3, 2),
                (1280, 1024, 1, 1),
            ]
            cfg['num_features'] = 1024
        else:
            cfg['head'] = [
                (416, 756, 3, 2),
                (756, 1024, 3, 1),
                (1024, 1024, 3, 2),
                (1024, 1280, 1, 1),
            ]
            cfg['num_features'] = 1280
    else:
        raise ValueError('Invalid net configuration ' + variant + ' !!!')

    model = TSelecSLSAA(cfg, **model_kwargs)
    model.default_cfg = default_cfgs[variant]
    if pretrained:
        load_pretrained(
            model,
            num_classes=model_kwargs.get('num_classes', 0),
            in_chans=model_kwargs.get('in_chans', 3),
            strict=True)
    return model


@register_model
def aa_tselecsls42(pretrained=False, **kwargs):
    """Constructs a aa_tselecSLS42 model.
    """
    return _create_model('aa_tselecsls42', pretrained, kwargs)


@register_model
def aa_tselecsls42b(pretrained=False, **kwargs):
    """Constructs a aa_tselecSLS42_B model.
    """
    return _create_model('aa_tselecsls42b', pretrained, kwargs)


@register_model
def aa_tselecsls60(pretrained=False, **kwargs):
    """Constructs a aa_tselecSLS60 model.
    """
    return _create_model('aa_tselecsls60', pretrained, kwargs)


@register_model
def aa_tselecsls60b(pretrained=False, **kwargs):
    """Constructs a aa_tselecSLS60_B model.
    """
    return _create_model('aa_tselecsls60b', pretrained, kwargs)

