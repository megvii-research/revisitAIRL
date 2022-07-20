#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch

from copy import deepcopy

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl

  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"

  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    all_dict = obj['model']
    all_dict = {k.replace('module.', ''): v for k, v in all_dict.items()}

    online_encoder_dict = {}
    target_encoder_dict = {}

    for k, v in all_dict.items():
        if k.startswith('online_encoder.'):
            online_encoder_dict[k.replace('online_encoder.', '')] = v
        elif k.startswith('encoder_q.'):
            online_encoder_dict[k.replace('encoder_q.', '')] = v
        elif k.startswith('student_encoder.'):
            online_encoder_dict[k.replace('student_encoder.', '')] = v
        elif k.startswith('target_encoder.'):
            target_encoder_dict[k.replace('target_encoder.', '')] = v
        elif k.startswith('encoder_k.'):
            target_encoder_dict[k.replace('encoder_k.', '')] = v
        elif k.startswith('teacher_encoder.'):
            target_encoder_dict[k.replace('teacher_encoder.', '')] = v
        else:
            print(k)

    if len(sys.argv) < 4:
        new_obj = online_encoder_dict
    else:
        if 'target' in sys.argv[-1]:
            new_obj = target_encoder_dict
        elif 'supervised' in sys.argv[-1]:
            new_obj = all_dict
        else:
            raise ValueError('Wrong argument.')

    obj = deepcopy(new_obj)
    newmodel = {}
    for k in list(obj.keys()):
        if 'fc' in k:
            continue
        old_k = k
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace("layer{}".format(t), "res{}".format(t + 1))
        for t in [1, 2, 3]:
            k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
        print('Convert finished.')
    if obj:
        print("Unconverted keys:", obj.keys())
