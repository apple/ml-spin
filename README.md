# SPIN
This repository contains the official implementation for the ECCV'23 paper, ["SPIN: An Empirical Evaluation on Sharing Parameters of Isotropic Networks"](https://arxiv.org/abs/2207.10237).

## Code Overview
We provide the implementation of weight sharing version of the [ConvMixer](https://openreview.net/pdf?id=TVHS5Y4dNvM) model. The main code for the implementation are in the `models` directory. The model can be configured by the files in `configs`. We provide three example configs.
* `configs/ConvMixer.yaml` for vanilla ConvMixer model.
* `configs/WS-ConvMixer.yaml` for Weight-shared ConvMixer (WS-ConvMixer) model.
* `configs/WFWS-ConvMixer.yaml` for Weight-fusion Weight-shared ConvMixer (WFWS-ConvMixer) model.

Note that in order to run the model `configs/WF-WSConvMixer.yaml`, you must have a corresponding pretrained ConvMixer model. Please refer to our paper for each technique.

## Installation
First, clone this repo with
```
git clone https://github.com/apple/ml-spin.git
```
The implementation of SPIN reuses the infrastructure of Meta Research's open source project [SlowFast](https://github.com/facebookresearch/SlowFast). Our modification to the SlowFast code is stored in the `spin-slowfast.patch`. To download the SlowFast code and apply our changes, run
```
bash setup.sh
```
After getting the codebase ready, follow this [link](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) from SlowFast repo to setup your environment and install other dependencies.

## Training
After the environment is set up, you can run the following example training script to train a weight sharing ConvMixer model. The script assumes you have a machine with 4-GPUs.
```
bash run.sh
```
### Pre-trained ConvMixer Models on ImageNet1K
We provide our pretrained models of ConvMixer, WS-ConvMixer and WFWS-ConvMixer in the following table. For the WFWS-ConvMixer, we first initialized the model using the proposed weight fusion technique with mean strategy, and then run the `models/fuse_weights.py` to export the fused model after training. In order to re-run the model, please use the WS-ConvMixer configuration. Please note we did a light hyperparameter tunning so the accuracy is slightly higher than the numbers reported in the paper.
| C/D/P/K | Weight Sharing? | Weight Fusion? | Sharing Rate | Share Distribution | Sharing Mapping | Accuracy | Model Size |
| ------- | --------------- | -------------- | ------------ | ------------- | -------------------- | -------- | ---------- |
| 768/32/14/3 | No  | No  | - | - | - | 76.32% | [79MB](pretrained/ConvMixer_768_32_14_3-Stripped.pyth)
| 768/32/14/3 | Yes | No  | 2 | Uniform | Sequential | 74.27% | [43MB](pretrained/WS-ConvMixer_768_32_14_3-Stripped.pyth) |
| 768/32/14/3 | Yes | Mean | 2 | Uniform | Sequential | 75.21%  | [43MB](pretrained/WF-Mean-WS-ConvMixer_768_32_14_3-Fused-Stripped.pyth) |

## Citation
If you find our code or paper helps, please consider citing:
```
@article{spin_eccv22,
    author    = {Lin, Chien-Yu and Prabhu, Anish and Merth, Thomas and Mehta, Sachin and Ranjan, Anurag and Horton, Maxwell and Rastegari, Mohammad}
    title     = {SPIN: An Empirical Evaluation on Sharing Parameters of Isotropic Networks},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2022}
}
```
