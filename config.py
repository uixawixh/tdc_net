# !/usr/bin/python
# coding:utf-8
import os
import random

import torch

# CPU worker to run
CPU_CORE = min(os.cpu_count() - 1, 1)
# Device for torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Random seed
SEED = 1007
# Atom radius(Ang.) to pixel
SCALE = 10
# Trained models path dict
MODEL_DICT = {
    'GAP_HSE_PBE': 'datasets/hse_set_pbe',
    'GAP_HSE': 'datasets/hse_set',
    'GAP_PBE': 'datasets/pbe_set',
}

random.seed(SEED)
