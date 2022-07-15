#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import slowfast.utils.weight_init_helper as init_helper
import torch
import torch.nn as nn
from convmixer_helper import ConvMixerHead, ConvMixerPatchEmbed, WSConvMixerBlock
from slowfast.models.build import MODEL_REGISTRY
from weight_sharing_helper import WeightSharingTopology


@MODEL_REGISTRY.register()
class WSConvMixer(nn.Module):
    """The main class of the weight sharing ConvMixer

    We support non-weight-sharing, weight-sharing and weight-sharing with pretrained weights
    all in this single module. The mode is configured by the cfg file feed to __init__().
    """

    def __init__(self, cfg):
        """The `__init__` method of the weight sharing ConvMixer

        Args:
            cfg (CfgNode): model building configs, details are in the comments of the config file.
        """
        super().__init__()
        channel = cfg.CONVMIXER.CHANNEL
        patch_kernel = cfg.CONVMIXER.PATCH_KERNEL
        patch_stride = cfg.CONVMIXER.PATCH_STRIDE
        patch_pad = cfg.CONVMIXER.PATCH_PADDING
        kernel = cfg.CONVMIXER.KERNEL
        act_func = cfg.CONVMIXER.ACT_FUNC
        classes = cfg.MODEL.NUM_CLASSES
        pad = self._cal_pad(kernel)
        self.depth = cfg.CONVMIXER.DEPTH

        # weight sharing module related parameters
        self.share_weight = cfg.CONVMIXER.WEIGHT_SHARE.ENABLE
        self.share_rate = cfg.CONVMIXER.WEIGHT_SHARE.SHARE_RATE
        self.num_share = int(self.depth // self.share_rate)
        self.share_dist = cfg.CONVMIXER.WEIGHT_SHARE.SHARING_DISTRIBUTION
        self.share_map = cfg.CONVMIXER.WEIGHT_SHARE.SHARING_MAPPING
        self.reduction_fn = cfg.CONVMIXER.WEIGHT_SHARE.REDUCTION_FN

        if act_func == "GELU":
            self.activation = nn.GELU
        elif act_func == "RELU":
            self.activation = nn.ReLU
        else:
            raise NotImplementedError(
                "{} is not supported as an activation function".format(act)
            )

        _SUPPORTED_REDUCTION_FNS = [
            "mean",
            "choose_first",
            "scalar_weighted_mean",
            "channel_weighted_mean",
        ]
        if self.reduction_fn not in _SUPPORTED_REDUCTION_FNS:
            raise NotImplementedError(
                "Only [mean, choose_first, channel_weighted_mean, scalar_weighted_mean] weight fusion strategies are supported."
            )

        if self.share_weight and self.reduction_fn == None:
            num_pwise_w = self.num_share
        else:
            num_pwise_w = self.depth

        self.pwise_w = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(channel, channel, 1, 1))
                for _ in range(num_pwise_w)
            ]
        )
        self.pwise_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(channel)) for _ in range(num_pwise_w)]
        )

        if self.reduction_fn == "scalar_weighted_mean":
            self.reduction_w = nn.ParameterList(
                [nn.Parameter(data=torch.ones(1)) for _ in range(self.depth)]
            )
        elif self.reduction_fn == "channel_weighted_mean":
            self.reduction_w = nn.ParameterList(
                [nn.Parameter(data=torch.ones(channel)) for _ in range(self.depth)]
            )

        # build sharing mapping
        self.pwise_map = self.build_layer_mapping()

        self.patch_embed = ConvMixerPatchEmbed(
            dim_in=3,
            dim_out=channel,
            kernel=patch_kernel,
            stride=patch_stride,
            activation=self.activation,
            padding=patch_pad,
        )

        self.blocks = nn.ModuleList()
        for _ in range(self.depth):
            self.blocks.append(
                WSConvMixerBlock(
                    dim=channel,
                    kernel_size=kernel,
                    padding=pad,
                    activation=self.activation,
                )
            )

        self.head = ConvMixerHead(
            dim=channel, dropout_rate=cfg.MODEL.DROPOUT_RATE, classes=classes
        )

        # init weights
        for w in self.pwise_w:
            nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
        init_helper.init_weights(self)

    def _cal_pad(self, kernel, stride=1):
        """Helper function to calculate padding size."""
        if kernel > stride:
            raise ValueError("size of kernel should be larger than stride")
        pad = int((kernel - stride) / 2)
        return pad

    def build_layer_mapping(self):
        """Function to define the sharing mapping."""
        if not self.share_weight:
            mapping = [i for i in range(self.depth)]
        else:
            mapping = getattr(
                WeightSharingTopology, f"{self.share_dist}_{self.share_map}"
            )(self.share_rate, self.num_share)
        assert self.depth == len(mapping), "Length of Mapping doesn't match model depth"
        return mapping

    def weight_fusion(self):
        """Generate weights from pretrained weights during forward."""
        w = self.pwise_w
        b = self.pwise_b

        reduced_w = []
        for i in range(self.num_share):
            if self.reduction_fn == "mean":
                p = torch.mean(
                    torch.stack(
                        [w[i * self.share_rate + j] for j in range(self.share_rate)]
                    ),
                    dim=0,
                )
            elif self.reduction_fn == "choose_first":
                p = w[i * self.share_rate]
            elif (
                self.reduction_fn == "channel_weighted_mean"
                or self.reduction_fn == "scalar_weighted_mean"
            ):
                tf_p = [
                    self.reduction_w[i * self.share_rate + j]
                    for j in range(self.share_rate)
                ]
                p = [w[i * self.share_rate + j] for j in range(self.share_rate)]
                p = [t.view(-1, 1, 1, 1) * x for t, x in zip(tf_p, p)]
                p = torch.mean(torch.stack(p), dim=0)
            reduced_w.append(p)

        reduced_b = []
        for i in range(self.num_share):
            if self.reduction_fn == "choose_first":
                p = b[i * self.share_rate]
            elif (
                self.reduction_fn == "mean"
                or self.reduction_fn == "scalar_weighted_mean"
                or self.reduction_fn == "channel_weighted_mean"
            ):
                p = torch.mean(
                    torch.stack(
                        [b[i * self.share_rate + j] for j in range(self.share_rate)]
                    ),
                    dim=0,
                )
            reduced_b.append(p)

        return reduced_w, reduced_b

    def forward(self, x):
        """The forward method of WSConvMixer module."""
        x = x[0]
        x = self.patch_embed(x)

        if self.reduction_fn is not None:
            w, b = self.weight_fusion()
        else:
            w, b = self.pwise_w, self.pwise_b

        for i in range(self.depth):
            ptr = self.pwise_map[i]
            x = self.blocks[i](x, w[ptr], b[ptr])

        x = self.head(x)
        return x
