#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import slowfast.utils.checkpoint as cu
import torch
import torch.nn as nn
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.models import build_model
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args


def fuse_weights(cfg):
    model = build_model(cfg)
    checkpoint = torch.load(cfg.TRAIN.CHECKPOINT_FILE_PATH)
    model.load_state_dict(checkpoint["model_state"])

    fused_pwise_w, fused_pwise_b = model.weight_fusion()
    num_fused_lay = len(fused_pwise_w)

    new_pwise_w = nn.ParameterList(
        [nn.Parameter(data=fused_pwise_w[i]) for i in range(num_fused_lay)]
    )
    new_pwise_b = nn.ParameterList(
        [nn.Parameter(data=fused_pwise_b[i]) for i in range(num_fused_lay)]
    )

    model.pwise_w = new_pwise_w
    model.pwise_b = new_pwise_b
    model.reduction_fn = None

    cu.save_checkpoint("./pretrained", model, None, 0, cfg, None)


def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    launch_job(cfg=cfg, init_method=args.init_method, func=fuse_weights)


if __name__ == "__main__":
    main()
