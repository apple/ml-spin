#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

PL_TORCH_DISTRIBUTED_BACKEND=gloo python SlowFast/tools/run_net.py \
  --cfg configs/WS-ConvMixer.yaml \
  DATA.PATH_TO_DATA_DIR /PATH/TO/IMAGENET/DATA/ \
  NUM_GPUS 4 \
  DATA_LOADER.NUM_WORKERS 4 \
  TRAIN.BATCH_SIZE 256 \
  TENSORBOARD.ENABLE True \
  TRAIN.ENABLE True \
  TEST.ENABLE False
