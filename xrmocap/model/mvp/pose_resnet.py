import logging
import torch.nn as nn
from mmcv.cnn.resnet import ResNet
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.utils.distribute_utils import is_main_process


class PoseResNet(nn.Module):
    """PoseResNet with ResNet as backbone."""

    def __init__(self,
                 logger: Union[None, str, logging.Logger],
                 n_kps: int,
                 deconv_with_bias: bool,
                 n_deconv_layers: int,
                 n_deconv_filters: list,
                 n_deconv_kernels: list,
                 final_conv_kernel: int,
                 inplanes: int = 2048,
                 n_layers: int = 50,
                 **kwargs):
        """Create the Pose Resnet backbone network.

        Args:
            logger (Union[None, str, logging.Logger]):
                Logger for logging. If None, root logger will be selected.
            n_kps (int):
                Number of keypoints.
            deconv_with_bias (bool):
                Whether to introduce bias in deconvolution.
            n_deconv_layers (int):
                Number of deconvolution layers.
            n_deconv_filters (list):
                Number of deconvolution filters.
            n_deconv_kernels (list):
                Number of deconvolution kernels.
            final_conv_kernel (int):
                Kernel size of the final convolution layer.
            inplanes (int, optional):
                Number of input channels. Defaults to 2048.
            n_layers (int, optional):
                Number of layers of the Resnet. Defaults to 50.
        """
        self.logger = get_logger(logger)
        self.inplanes = inplanes
        self.deconv_with_bias = deconv_with_bias

        super(PoseResNet, self).__init__()

        # resnet backbone for PoseResNet
        self.resnet = ResNet(depth=n_layers)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            n_deconv_layers,
            n_deconv_filters,
            n_deconv_kernels,
        )

        self.final_layer = nn.Conv2d(
            in_channels=n_deconv_filters[-1],
            out_channels=n_kps,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, n_layers, n_filters, n_kernels):
        assert n_layers == len(n_filters), \
            'ERROR: n_deconv_layers is different len(n_deconv_filters)'
        assert n_layers == len(n_kernels), \
            'ERROR: n_deconv_layers is different len(n_deconv_filters)'

        layers = []
        for i in range(n_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(n_kernels[i], i)

            planes = n_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, use_feat_level=[0, 1, 2]):
        x = self.resnet(x)[-1]

        interm_feat = []
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                interm_feat.append(x)

        return [f for (i, f) in enumerate(interm_feat) if i in use_feat_level]

    def init_weights(self):
        if is_main_process():
            self.logger.info('Backbone: Init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
